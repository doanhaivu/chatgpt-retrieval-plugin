from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

# Assuming OpenAI and Document are defined elsewhere
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.messages.ai import AIMessage

class TopicData(BaseModel):
    page_content: str

class Stage1Output(BaseModel):
    title: str
    rewrite: str

class TranscriptElement(BaseModel):
    text: str
    start: float
    duration: float

class SummarizeInput(BaseModel):
    transcript: List[TranscriptElement]

class SummaryOutput(BaseModel):
    Title: str
    Rewrite: str

class SummarizeOutput(BaseModel):
    topic_summary_items: List[SummaryOutput]
    video_summary: str

MODEL_NAME = 'gpt-4'

def create_sentences(segments, MIN_WORDS, MAX_WORDS):

  # Combine the non-sentences together
  sentences = []

  is_new_sentence = True
  sentence_length = 0
  sentence_num = 0
  sentence_segments = []

  for i in range(len(segments)):
    if is_new_sentence == True:
      is_new_sentence = False
    # Append the segment
    sentence_segments.append(segments[i])
    segment_words = segments[i].split(' ')
    sentence_length += len(segment_words)
    
    # If exceed MAX_WORDS, then stop at the end of the segment
    # Only consider it a sentence if the length is at least MIN_WORDS
    if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
      sentence = ' '.join(sentence_segments)
      sentences.append({
        'sentence_num': sentence_num,
        'text': sentence,
        'sentence_length': sentence_length
      })
      # Reset
      is_new_sentence = True
      sentence_length = 0
      sentence_segments = []
      sentence_num += 1

  return sentences

def create_chunks(sentences, CHUNK_LENGTH, STRIDE):
    chunks = []
    num_sentences = len(sentences)

    for i in range(0, num_sentences, CHUNK_LENGTH - STRIDE):
        chunk_text = []
        num_words = 0

        # Build the chunk and count words simultaneously
        for sentence in sentences[i:i+CHUNK_LENGTH]:
            chunk_text.append(sentence['text'])
            num_words += sentence['sentence_length']

        # Combine the text of the sentences into one string
        combined_text = ' '.join(chunk_text)
        
        # Create a chunk dictionary
        chunks.append({
            'start_sentence_num': sentences[i]['sentence_num'],
            'end_sentence_num': sentences[min(i + CHUNK_LENGTH - 1, num_sentences - 1)]['sentence_num'],
            'text': combined_text,
            'num_words': num_words
        })

    return chunks

def parse_title_summary_results(results):
    out = []
    for e in results:
        # Removing newline characters for easier processing
        e = e.replace('\n', '')
        
        title = ''
        summary = ''
        
        if 'Title:' in e and 'Rewrite:' in e:
            title_part = e.split('Rewrite:')[0]
            summary_part = e.split('Rewrite:')[1]
            
            if 'Title:' in title_part:
                title = title_part.split('Title:')[1].strip()
            
            summary = summary_part.strip()

        processed = {
            'Title': title,
            'Rewrite': summary
        }
        out.append(processed)
    return out

def print_ai_messages(messages: List[AIMessage]):
    for i, message in enumerate(messages, start=1):
        print(f"{i} {message.content.split('Rewrite:')[0].strip()}")
        print(f"\n{message.content.split('Rewrite:')[1].strip()}")
        print("-" * 50)
        print()

def summarize_stage_1(chunks_text):

  # Prompt to get title and summary for each chunk
  map_prompt_template = """
  Firstly, give the below text an informative title.
  Then, on a new line, rewrite the text by converting it from spoken language to written language,
  Keep it concise while retaining the original wording, facts, and examples as much as possible
  Use direct language.
  You must follow the rules:
    - The rewrite should be as detailed as needed to make the summary comprehensive.
    - Do not mention the author or speaker at all should act as an independent writing.
    - Do not use any of sentence starters or transitional phrases such as: The text outlines, It emphasizes, The text advises, It suggests, The text concludes, The text emphasizes, The text provides, The text discusses...
  Return your answer in the following format:

  Title:
  Rewrite:

  {text}
  """

  map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

  # Define the LLMs
  map_llm = ChatOpenAI(temperature=0, model_name = MODEL_NAME, max_tokens=None, timeout=None, max_retries=2)

  map_llm_chain = map_prompt | map_llm
  map_llm_chain_input = [{'text': t} for t in chunks_text]
  # Run the input through the LLM chain (works in parallel)
  map_llm_chain_results = map_llm_chain.batch(map_llm_chain_input)

  if not isinstance(map_llm_chain_results, list):
    stage_1_outputs = parse_title_summary_results([map_llm_chain_results.content])
  else:
    stage_1_outputs = parse_title_summary_results([e.content for e in map_llm_chain_results])

  return {
    'stage_1_outputs': stage_1_outputs
  }


def get_topics(title_similarity, num_topics=8, bonus_constant=0.25, min_size=3):
  num_chunks = title_similarity.shape[0]

  # Vectorized calculation of proximity bonuses
  row_indices, col_indices = np.indices((num_chunks, num_chunks))
  epsilon = 1e-10  # Small constant to prevent division by zero
  proximity_bonus_arr = np.where(row_indices != col_indices, 
                                  bonus_constant / (np.abs(row_indices - col_indices) + epsilon), 
                                  0)
  
  # Adding proximity bonus to title similarity matrix
  title_similarity += proximity_bonus_arr

  # Create the network graph from the similarity matrix
  title_nx_graph = nx.from_numpy_array(title_similarity)

  # Initializing variables for community detection
  desired_num_topics = num_topics
  topics_title_accepted = []

  # Finding the initial suitable resolution
  resolution = 0.80
  resolution_step = 0.005

  # Initial search for a suitable resolution
  topics_title = []
  while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
    topics_title = community.louvain_communities(title_nx_graph, weight='weight', resolution=resolution)
    resolution += resolution_step

  # Calculate the standard deviation of topic sizes for the initial partition
  topic_sizes = [len(c) for c in topics_title if len(c) >= min_size]  # Exclude small communities
  sizes_sd = np.std(topic_sizes) if topic_sizes else float('inf')

  # Store the best partitioning based on standard deviation
  lowest_sd_iteration = 0
  lowest_sd = sizes_sd
  
  # Main loop to find the best partition
  iterations = 40
  for i in range(iterations):
    topics_title = community.louvain_communities(title_nx_graph, weight='weight', resolution=resolution)

    topic_sizes = [len(c) for c in topics_title if len(c) >= min_size]  # Exclude small communities
    sizes_sd = np.std(topic_sizes) if topic_sizes else float('inf')

    # Store the current partition
    topics_title_accepted.append(topics_title)

    # Check if the current partition meets the criteria
    if sizes_sd < lowest_sd:
      lowest_sd_iteration = i
      lowest_sd = sizes_sd

    # Increment resolution for the next iteration
    resolution += resolution_step

  # Use the best partition found
  topics_title = topics_title_accepted[lowest_sd_iteration]
  print(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

  # Order topics based on their mean indices
  topic_id_means = [np.mean(list(c)) for c in topics_title]  # Convert set to list for mean calculation
  topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key=lambda pair: pair[0])]
  
  # Assign each chunk to a topic
  chunk_topics = [None] * num_chunks
  for i, c in enumerate(topics_title):
    for j in c:
      chunk_topics[j] = i
            
  return {
    'chunk_topics': chunk_topics,
    'topics': topics_title
  }

def summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250):

  # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
  title_prompt_template = """
  Write an informative title that summarizes each of the following groups of titles.
  Make sure that the titles capture as much information as possible, and are different from each other.
  Use direct language.
  You must follow the rules:
    - Do not mention the author or speaker at all should act as an independent writing.
    - Do not use any of sentence starters or transitional phrases such as: The text outlines, It emphasizes, The text advises, It suggests, The text concludes, The text emphasizes, The text provides, The text discusses...
  Return your answer in a list, with new line separating each title:

  {text}
  """

  map_prompt_template = """
  Write a 75-100 word summary of the following text.
  Keep it concise while retaining the original wording, facts, and examples as much as possible.
  Use direct language.
  You must follow the following rules:
    - The summary should be as detailed as needed to make the summary comprehensive.
    - Do not mention the author or speaker at all should act as an independent writing.
    - Do not use any of sentence starters or transitional phrases such as: The text outlines, It emphasizes, The text advises, It suggests, The text concludes, The text emphasizes, The text provides, The text discusses...

  {text}
  """

  combine_prompt_template = 'Write a ' + str(summary_num_words) + """-word summary of the following, removing irrelevant information.
  Keep it concise while retaining the original wording, facts, and examples as much as possible.
  Use direct language.
  You must follow the following rules:
    - The summary should be as detailed as needed to make the summary comprehensive.
    - Do not mention the author or speaker at all should act as an independent writing.
    - Do not use any of sentence starters or transitional phrases such as: The text outlines, It emphasizes, The text advises, It suggests, The text concludes, The text emphasizes, The text provides, The text discusses...
  
  {text}
  """

  title_prompt = PromptTemplate(template=title_prompt_template, input_variables=["text"])
  map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
  combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

  topics_data = []
  for c in topics:
    topic_data = {
      'rewrites': [stage_1_outputs[chunk_id]['Rewrite'] for chunk_id in c],
      'titles': [stage_1_outputs[chunk_id]['Title'] for chunk_id in c]
    }
    topic_data['rewrites_concat'] = ' '.join(topic_data['rewrites'])
    topic_data['titles_concat'] = ', '.join(topic_data['titles'])
    topics_data.append(topic_data)
    
  # Get a list of each community's summaries (concatenated)
  topics_summary_concat = [c['rewrites_concat'] for c in topics_data]
  topics_titles_concat = [c['titles_concat'] for c in topics_data]

  # Concat into one long string to do the topic title creation
  topics_titles_concat_all = ''''''
  for i, c in enumerate(topics_titles_concat):
    topics_titles_concat_all += f'''{i+1}. {c}
    '''
  
  title_llm = ChatOpenAI(temperature=0, model_name = MODEL_NAME, max_tokens=None, timeout=None, max_retries=2)
  title_llm_chain = title_prompt | title_llm
  
  title_llm_chain_input = [{'text': topics_titles_concat_all}]
  title_llm_chain_results = title_llm_chain.batch(title_llm_chain_input)

  if isinstance(title_llm_chain_results, list):
      contents = [msg.content for msg in title_llm_chain_results]
      all_content = '\n'.join(contents)
  else:
      # If it's not a list, assume it's a single AIMessage object and get the content directly
      all_content = title_llm_chain_results.content

  # Now split the combined content by newlines
  titles = all_content.split('\n')

  # Remove any empty titles
  titles = [t for t in titles if t != '']
  # Remove spaces at start or end of each title
  titles = [t.strip() for t in titles]

  print('========summarize_stage_2 titles=========')
  for title in titles:
    print(title)

  map_llm = ChatOpenAI(temperature=0, model_name = MODEL_NAME, max_tokens=None, timeout=None, max_retries=2)
  reduce_llm = ChatOpenAI(temperature=0, model_name = MODEL_NAME, max_tokens=None, timeout=None, max_retries=2)

  # Run the map-reduce chain
  docs = [Document(page_content=t) for t in topics_summary_concat]
  chain = load_summarize_chain(chain_type="map_reduce", map_prompt = map_prompt, combine_prompt = combine_prompt, return_intermediate_steps = True,
                              llm = map_llm, reduce_llm = reduce_llm)

  output = chain({"input_documents": docs}, return_only_outputs = True)
  summaries = output['intermediate_steps']

  print('========summarize_stage_2 summaries=========')
  for aSummary in summaries:
    print(aSummary)

  stage_2_outputs = [{'Title': t, 'Rewrite': s} for t, s in zip(titles, summaries)]
  final_summary = output['output_text']

  out = {
    'topic_summary_items': stage_2_outputs,
    'video_summary': final_summary
  }
  
  return out
