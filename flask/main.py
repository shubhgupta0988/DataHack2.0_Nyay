import os
from flask import Flask, request, jsonify
import fasttext
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
from flask_cors import CORS
import tensorflow as tf
import google.generativeai as palm
from deep_translator import GoogleTranslator
import re
palm.configure(api_key="AIzaSyAmJZX92dPwWWwyWRyPzWukcmbawQiAzGg")

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0.7,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
}

import detectlanguage

detectlanguage.configuration.api_key = "f9455d42299792e22ec506a2a638f3ed"

data = pd.read_csv("./data.csv")
model = fasttext.load_model('fasttext.bin')
texts = []

LSTM_model = tf.keras.saving.load_model('model.h5')
print('Loaded Model')

def predict(name):
    print("Name = ", name)
    name_samplevector = cv.transform([name]).toarray()
    prediction = LSTM_model.predict([name_samplevector])

    if prediction >= 0.5:
        return "male"
    else:
        return "female"

            
for item in data['sentence']:
  texts.append(item)

text_data = texts
vector_data = [model.get_sentence_vector(text) for text in text_data]

input_file = 'input.txt'

from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv('./Gender_Data.csv')
cv=CountVectorizer(analyzer='char')
X =list( dataset['Name'])
X=cv.fit_transform(X).toarray()

# Create and build an Annoy index
def build_annoy_index(vector_data, index_filename):
    t = AnnoyIndex(len(vector_data[0]), metric='euclidean')
    for i, vector in enumerate(vector_data):
        t.add_item(i, vector)
    t.build(1000)  # Adjust the number of trees as needed
    t.save(index_filename)

index_filename = 'annoy_index.ann'
build_annoy_index(vector_data, index_filename)


app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query_lawyers():
    if request.method == 'POST':
        res = request.get_json()

        # Get the user's query from the POST request
        user_prompt = res.get('query')
        dl = detectlanguage.detect(user_prompt)
        src = dl[0]['language']
        text_chunks = [user_prompt[i:i+4999] for i in range(0, len(user_prompt), 4999)]

        # Initialize an empty list to store translated chunks
        translated_chunks = []

        # Translate each chunk and store the results
        for chunk in text_chunks:
            translated_chunk = GoogleTranslator(source=src, target='en').translate(chunk)
            translated_chunks.append(translated_chunk)

        # Concatenate the translated chunks to get the final translated text
        user_prompt = " ".join(translated_chunks)

        prompt2 = f"""Any case can be classified into one of the following types only: Banking and Finance, Intellectual Property, Media and Entertainment, Real Estate, Human Rights, Labor Law, Civil, Constitutional, Corporate, Criminal, Environmental, Family, Immigration, Medical, Consumer Protection, Tax. I need you to classify the user concern or requirement into one of the given types and that is the 'domain'. The user input is : {user_prompt}. Also find if the location of the laawyer is given and the language requirements are given. Your output format is 'The domain required is <domain>, the languages required are <languages> and the location is <location>.' Replace <> with the values received from the user's prompt if it exists otherwise put N/A in its place."""
        
        response = palm.generate_text(
        **defaults,
        prompt=prompt2
        )

        print(response.result)
        generated_text = response.result

        user_embeddings = model.get_sentence_vector(generated_text)
        # print(user_embeddings)
        sentences = data["sentence"]

        embeddings = []

        for sentence in sentences:
            embed = model.get_sentence_vector(sentence)
            embeddings.append(embed)
        # print(embeddings[0])
        from sklearn.metrics.pairwise import cosine_similarity
        # Ensure that user_embeddings and lawyer_embeddings have the same dimension
        print("User embeddings shape:", user_embeddings.shape)
        print("Number of lawyer embeddings:", len(embeddings))

        # Calculate cosine similarity for each lawyer_embedding
        similarities = [cosine_similarity(np.array([user_embeddings]), np.array([lawyer_embedding])) for lawyer_embedding in embeddings]

        # Check the similarities list
        # print("Similarities:", similarities)
        # print("Hello")
        lawyers_with_gender = [{'similarity': sim, 'gender': sentence.split("Gender is ")[-1], 'info': sentence} for [[sim]], sentence in zip(similarities, data['sentence'])]
        lawyers_with_gender.sort(key=lambda x: x['similarity'], reverse=True)
        # print(lawyers_with_gender)
        male_count = 0
        female_count = 0

        top_lawyers = []


        for lawyer in lawyers_with_gender:
            if lawyer['gender'] == 'male.' and male_count < 5:
                top_lawyers.append(lawyer)
                male_count += 1
            elif lawyer['gender'] == 'female.' and female_count < 5:
                top_lawyers.append(lawyer)
                female_count += 1
            if male_count == 5 and female_count == 5:
                break
        
        top_lawyers_df = pd.DataFrame(top_lawyers)

        names=[]
        experiences = []
        categories = []
        ratings = []
        languages = []
        locations = []

        for d in top_lawyers_df["info"]:
            name_pattern = r'^(?:\w+\s+){2}'
            experience_pattern = r'(?<=has\s)\d+(?=\syear)'
            category_pattern = r'(?<=years of experience in\s)(.*?)(?=\.)'
            ratings_pattern = r'(?<=He has a Client Feedback of\s)(.*?)(?=\.)'
            language_pattern = r'(?<=He speaks:\s)(.*?)(?=\.)'
            location_pattern = r'(?<=based in\s)(.*?)(?=\.)'

            # Extract name
            name = re.search(name_pattern, d).group().strip()
            names.append(name)

            # Extract years of experience
            experience = int(re.search(experience_pattern, d).group())
            experiences.append(experience)

            # Extract category
            category = re.search(category_pattern, d).group().strip()
            categories.append(category)

            # Extract ratings
            ratings_value = re.search(ratings_pattern, d).group().strip()
            ratings.append(ratings_value)

            language = re.search(language_pattern, d).group().strip()
            languages.append(language)

            location = re.search(location_pattern, d).group().strip()
            locations.append(location)

        # Create a DataFrame from the extracted lists
            dt = {
                "Name": names,
                "Exp": experiences,
                "Category": categories,
                "Ratings": ratings,
                "Language": languages,
                "Location": locations,
                "XAI": generated_text
            }

            lawyers_df = pd.DataFrame(dt)

# Print the resulting DataFrame
        print(lawyers_df)

        top_lawyers_json = lawyers_df.to_json(orient='records')

        #print(top_lawyers_df)
        return top_lawyers_json


@app.route('/change', methods=['POST'])
def change():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'})
    
        file = request.files['file']
        if file.filename.lower().endswith(('.xls', '.xlsx')):
            try:
                # Read Excel file
                df = pd.read_excel(file, engine='openpyxl')
            except Exception as e:
                return jsonify({'error': f'Error reading Excel file: {str(e)}'})
        elif file.filename.lower().endswith('.csv'):
            try:
                # Read CSV file
                df = pd.read_csv(file, index_col=False)
            except Exception as e:
                return jsonify({'error': f'Error reading CSV file: {str(e)}'})
        else:
            return jsonify({'error': 'Unsupported file format'})
        print(df['Name'])
        df['name'] = df['Name']
        df['experience'] = df['Information'].str.extract(r'(\d+) years of experience')
        df['languages'] = df['Information'].str.extract(r'(He speaks: (.+?)|is proficient in (.+?))\.')[1]
        df['languages'] = df['languages'].fillna('')
        import re

        def extract_unique_languages(row):
            languages = re.findall(r'\b\w+\b', row['languages'].lower())  # Replace NaN with empty string
            unique_languages = set()
            for lang in languages:
                if lang != 'and':
                    unique_languages.add(lang)
            return ', '.join(sorted(unique_languages))

        # Apply the function to all rows
        df['languages'] = df.apply(extract_unique_languages, axis=1)

        df['domains'] = df['Information'].str.extract(r'experience in (.+?)\.')
        df['domains'] = df['domains'].str.replace(',', '').str.replace(' and ', ' ')

        extracted_data = df[[ 'domains' ]]

        df['domains'] = extracted_data['domains']

        df['feedback'] = df['Information'].str.extract(r'Client Feedback of (.+?) out of 5.0')

        df['jurisdiction'] = df['Information'].str.extract(r'Jurisdiction is (.+?)\.')

        df['charge'] = df['Information'].str.extract(r'charges (.+?) USD')

        df['disposal_days'] = df['Information'].str.extract(r'takes (.+?) Avg Days for Disposal')

        df['firm'] = df['Information'].str.extract(r'practices at (.+?),')

        df['location'] = df['Information'].str.extract(r'based in (.+?)\.')

        df['pro bono'] = df['Information'].str.extract(r'(\w+)\s+(?:provides?|provide)')
        df['pro bono'] = df['pro bono'].apply(lambda x: 'no' if x == 'not' else 'yes')

        df['demographics'] = df['Information'].str.extract(r'Demographics is (.+?)\.')

        df = df[['name', 'experience', 'domains', 'feedback', 'languages', 'jurisdiction', 'charge', 'disposal_days', 'firm', 'location', 'pro bono', 'demographics']]

        df['gender'] = df['name'].apply(predict)

        # # Drop the 'first_name' column if you no longer need it
        # df.drop('name', axis=1, inplace=True)

        # Define a function to generate the sentence
        def generate_sentence(row):
            sentence = f"{row['name']} has {row['experience']} years of experience in {row['domains']}. "
            sentence += f"He has a Client Feedback of {row['feedback']} out of 5.0. "
            sentence += f"His Jurisdiction is {row['jurisdiction']}. "
            sentence += f"He charges {row['charge']} USD per hour. "
            sentence += f"He takes {row['disposal_days']} Avg Days for Disposal. "
            sentence += f"He speaks: {row['languages']}. "
            sentence += f"He practices at {row['firm']}, and is based in {row['location']}. "
            sentence += f"He does not provide pro bono services to the community. "
            sentence += f"His Client Demographics is {row['demographics']}."
            sentence += f"Gender is {row['gender']}."
            return sentence

        # Apply the function to each row in the DataFrame and store the generated sentences
        df['sentence'] = df.apply(generate_sentence, axis=1)

        df.to_csv("new.csv", index=False)

        folder_path = "./"

        # Filter out non-CSV files
        csv_files = ["data.csv", "new.csv"]

        # Create a list to hold the dataframes
        df_list = []

        for csv in csv_files:
            file_path = os.path.join(folder_path, csv)
            try:
                # Try reading the file using default UTF-8 encoding
                df = pd.read_csv(file_path)
                df_list.append(df)
            except UnicodeDecodeError:
                try:
                    # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
                    df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
                    df_list.append(df)
                except Exception as e:
                    print(f"Could not read file {csv} because of error: {e}")
            except Exception as e:
                print(f"Could not read file {csv} because of error: {e}")

        # Concatenate all data into one DataFrame
        big_df = pd.concat(df_list, ignore_index=True)

        # Save the final result to a new CSV file
        big_df.to_csv(os.path.join(folder_path, 'data.csv'), index=False)

        texts = []

        for item in big_df['sentence']:
            texts.append(item)
        
        input_file = 'input.txt'

        # Convert the DataFrame column to a list and save it to a text file with each row on a new line
        column_data = big_df['sentence'].tolist()
        with open(input_file, 'w') as file:  # Use a different variable name for the file object
            file.write('\n'.join(column_data))

        output_model = 'model.bin'
        model = fasttext.train_unsupervised(input=input_file, model='skipgram', lr=0.1, dim=100)

        # Save the trained model
        model.save_model(output_model)
        model.save_model('fasttext.bin')

        text_data = texts
        vector_data = [model.get_sentence_vector(text) for text in text_data]

        t = AnnoyIndex(len(vector_data[0]), metric='euclidean')

        # Create and build an Annoy index
        def build_annoy_index(vector_data, index_filename):
            t = AnnoyIndex(len(vector_data[0]), metric='euclidean')
            for i, vector in enumerate(vector_data):
                t.add_item(i, vector)
                print(f"Item {i}: {vector}")
            t.build(1000)  # Adjust the number of trees as needed
            t.save(index_filename)

        index_filename = 'annoy_index.ann'
        build_annoy_index(vector_data, index_filename)

        for i in range(t.get_n_items()):
            vector = t.get_item_vector(i)
            print(f"Item {i}: {vector}")
        return jsonify({'results': 'Embedding added successfully'})

        
if __name__ == '__main__':
    app.run()
