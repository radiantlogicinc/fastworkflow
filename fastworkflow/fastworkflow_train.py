import os
from typing import Optional
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import json
import os
from torch.utils.data import random_split
import fastworkflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple,Union
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=None
label_encoder=LabelEncoder()

def save_label_encoder(filepath):
    global label_encoder
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_label_encoder(filepath):
    global label_encoder
    with open(filepath, 'rb') as f:
        label_encoder = pickle.load(f)


# After training loop is complete
def save_model(model,tokenizer, save_path):
    # Save the model
    model.save_pretrained(save_path)
    # Save the tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")



def get_route_layer_filepath(workflow_folderpath) -> RouteLayer:
    command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory
    return os.path.join(
        cmddir.get_commandinfo_folderpath(workflow_folderpath),
        "tinymodel.pth"
    )

def get_route_layer_filepath1(workflow_folderpath) -> RouteLayer:
    command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory
    return os.path.join(
        cmddir.get_commandinfo_folderpath(workflow_folderpath),
        "largemodel.pth"
    )

def get_route_layer_filepath2(workflow_folderpath) -> RouteLayer:
    command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory
    return os.path.join(
        cmddir.get_commandinfo_folderpath(workflow_folderpath),
        "label_encoder.pkl"
    )


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_ndcg = 0
    num_batches = 0

    with torch.no_grad():
        for encodings, labels in tqdm(data_loader, desc="Evaluating"):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Calculate test loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            
            
            num_batches += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches  # Average test loss
    f1 = f1_score(all_labels, all_preds, average='weighted')
    

    return f1, avg_loss


def train_fastworkflows(session: fastworkflow.Session):

    workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
    command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory

    routes = []
    utterance_command_tuples = []
    for command_key in cmddir.get_utterance_keys():
        utterance_metadata = cmddir.get_utterance_metadata(command_key)
        utterances_func = utterance_metadata.get_generated_utterances_func(
                workflow_folderpath
            )
        utterance_list = utterances_func(session)

        command_name = cmddir.get_command_name(command_key)
        routes.append(Route(name=command_name, utterances=utterance_list))

        # dataset for training
        utterance_command_tuples.extend(
            list(zip(utterance_list, [command_name] * len(utterance_list)))
        )

        #rl = RouteLayer(encoder=self._encoder, routes=routes)

        # unpack the test data and train data
        X, y = zip(*utterance_command_tuples)
        num= len(set(y))
    

    model_name = "prajjwal1/bert-tiny"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tiny_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num).to(device)


    model_name = "distilbert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #large_model = AutoModel.from_pretrained(model_name).to(device)
    large_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num).to(device)
    global label_encoder
    dataset = list(zip(X, y))
    #label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Now create the dataset with encoded labels
    dataset = list(zip(X, y_encoded))
    train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

    def collate_batch(batch_texts, batch_labels):
        encodings = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True,
            max_length=128, 
            return_tensors='pt'
        )
        # Convert labels to long tensor
        labels = torch.tensor(batch_labels, dtype=torch.long)
        
        return encodings, labels

    train_loader = DataLoader(
        train_data,
        batch_size=10,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(
            [item[0] for item in batch],
            [item[1] for item in batch]  # Now these are numbers, not strings
        )
    )

    test_loader = DataLoader(
        test_data,
        batch_size=10,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            [item[0] for item in batch],
            [item[1] for item in batch]
        )
    )

    #batch_size = 64  # Increased batch size
    optimizer = AdamW(tiny_model.parameters(), lr=1e-4)  # Slightly higher learning rate
    num_epochs = 12

    path=get_route_layer_filepath1(workflow_folderpath)
    from time import time
    print("Starting training...")
    tiny_model.train()
    best_ndcg = 0
    best_f1 = 0
    training_start_time = time()
    training_losses = []  # Store training loss for each epoch
    test_losses = []
    for epoch in range(num_epochs):
        epoch_start_time = time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch_idx, (encodings, labels) in enumerate(progress_bar):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = tiny_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)  # Append training loss for the epoch

        # Evaluate after each epoch
        f1, avg_test_loss = evaluate_model(tiny_model, test_loader, device)
        test_losses.append(avg_test_loss)
        epoch_time = time() - epoch_start_time
        print(f"Epoch {epoch + 1} Results:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")
    
    path=get_route_layer_filepath(workflow_folderpath)
    save_model(tiny_model,tokenizer, path)   
    total_training_time = time() - training_start_time


    optimizer = AdamW(large_model.parameters(), lr=5e-5)
    num_epochs = 5

    print("Started training distilBert...")
    large_model.train()
    best_ndcg = 0
    best_f1 = 0
    num_epochs=5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch_idx, (encodings, labels) in enumerate(progress_bar):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = large_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

        # Evaluate after each epoch
        f1, avg_loss= evaluate_model(large_model, test_loader, device)
        print(f"Epoch {epoch + 1} Results:")
        print(f"F1 Score: {f1:.4f}")
       

    path=get_route_layer_filepath1(workflow_folderpath)
    save_model(large_model,tokenizer,path)  
    path=get_route_layer_filepath2(workflow_folderpath)
    save_label_encoder(path)


def predict_label(sentence, model_path):
    """
    Load a trained tiny BERT model and make predictions on a single sentence.
    
    Args:
        sentence (str): The input sentence to classify
        model_path (str): Path to the saved model directory
        label_encoder_path (str): Path to the saved label encoder
    
    Returns:
        str: The predicted label name
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    
    # Load the label encoder
    global label_encoder
    #path=get_route_layer_filepath2(path)
    path="./fastworkflow/_workflows/parameter_extraction/___command_info/label_encoder.pkl"
    load_label_encoder(path)
    
    # Tokenize the input sentence
    encodings = tokenizer(
        sentence, 
        padding=True, 
        truncation=True,
        max_length=128, 
        return_tensors='pt'
    )
    
    # Move tensors to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get predicted class and its confidence
        prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
        confidence = probabilities[0, prediction].item()
        
    # Convert numeric prediction back to label name
    label_name = label_encoder.inverse_transform([prediction])[0]
    
    return label_name, confidence