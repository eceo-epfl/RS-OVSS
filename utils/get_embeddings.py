import spacy
import torch
import nltk
nltk.download('wordnet')
nltk.download('stopwords')




def get_embeddings_from_clip (words, average_hidden_state=False):

    
    from transformers import AutoTokenizer, CLIPTextModel
    with torch.no_grad() :
        # Get only the clip text model : 
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        print()
        print('Getting clip embedding from CLIP....')
   
        inputs = tokenizer( words, 
                           padding=True, 
                           return_tensors="pt",
                           truncation=True,
                           max_length=77,
                           )
        outputs = model(**inputs)
        
        if average_hidden_state == False: 
            # use pooled output : return EOS token state
            # this returns the classification token 
            pooled_output = outputs.pooler_output  # pooled (EOS token) states
        
            # Normalize the output vectors : 
            embeddings = pooled_output.T / torch.norm(pooled_output, dim=1) 
            print('... normalised_pooled_output')
        else: 
            last_hidden_state = outputs.last_hidden_state
            embeddings = last_hidden_state.mean(dim=1)
            embeddings = embeddings.T / torch.norm(embeddings, dim=1)
            
            print('.... average hidden states')
            

    # Get similarity : 
    similarity_matrix = torch.einsum('ij,jk -> ik', embeddings.T, embeddings )    
    
    
    return embeddings, similarity_matrix





def get_embeddings_from_sentenceBERT(sentences):
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).t()

    print("Sentence embeddings shape:")
    print(sentence_embeddings.shape)

    # Get similarity : 
    similarity_matrix = torch.einsum('ij,jk -> ik', sentence_embeddings.t(), sentence_embeddings )    
    
    
    return sentence_embeddings, similarity_matrix





def get_embeddings_from_glove(words, return_similarity_matrix =True):
    
    nlp = spacy.load("en_core_web_lg") 
    embeddings = torch.zeros(size = [  300,len(words),])

    for idx, word in enumerate (words) :
        obj = nlp(word)
        normalised_vector   = obj.vector/ obj.vector_norm 
       # normalised_vector = obj.vector
        embeddings[:,idx] = torch.from_numpy(normalised_vector)
        
    similarity_matrix = torch.einsum('ij,jk -> ik', embeddings.T, embeddings )  
    
    if not return_similarity_matrix:
        return embeddings
     
    return embeddings, similarity_matrix

    return output
      
    



