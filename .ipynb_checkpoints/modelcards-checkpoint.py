import transformers
import torch
import intel_extension_for_pytorch as ipex
import random
import logging
import traceback
#Loading the model and tokenizer and putting it to the correct device


# def __init__(
#     self,
#     hf_model: str = 'Intel/neural-chat-7b-v3-3'
# ) -> None:
#     '''
#     Initializer for the ModelCardLLM class.

#     Arguments:
#     - hf_model: The string for the path to the Hugging Face model on the Model Hub. For example, if the full URL is 
#     https://huggingface.co/Intel/neural-chat-7b-v3-3, the string would be "Intel/neural-chat-7b-v3-3".
    
#     '''


def fetch_device():
    '''
    Returning the torch device as XPU if the Intel Max GPU is available; otherwise, return the CPU as the device.
    '''
    #Setting the PyTorch device. Intel Extension for PyTorch must be installed for this torch.xpu function to work.
    if torch.xpu.is_available(): # if using Intel Max GPU
        print('XPU is available')
        device = torch.device('xpu')
    else: #else use the Intel Xeon CPU
        print('XPU is not available. Returning device as CPU')
        device = torch.device('cpu')       
    return device

def check_device():
    '''
    Check if the XPU was targeted as a device. If so, set the device to the CPU to try that as well.
    If not, just return None
    '''
    if self.device.type.lower() == 'xpu':
        self.device.type = 'cpu'
        return self.device
    elif self.device.type.lower() == 'cpu':
        return None

def generate_response(
    hf_model, 
    device,
    system_input,
    user_input):
    
    '''
    A function to take a textual prompt, encode it into tokens, and generate new text based on the prompt.

    Arguments:
        hf_model: The Transformers loaded model
        tokenizer: The Transformers tokenizer
        system_input (str): A string of English instructions on what the model should do
        user_input (str): A follow-up string that is concatenated to the system_input to 
            ask a more specific question or make a statement to complete by the model.
    '''
    
    model = transformers.AutoModelForCausalLM.from_pretrained(hf_model).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
    
    # Format the input using the provided template
    prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"

    # Tokenize and encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Set pad_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    
    # Generate a response
    outputs = model.generate(inputs,
                             max_length=1000, 
                             num_return_sequences=1,
                            pad_token_id = tokenizer.pad_token_id)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response_assistant_only = response.split("### Assistant:\n")[-1]
    
    print(response_assistant_only)
    # expected response for default prompt
    # To calculate the sum of 100, 520, and 60, we will follow these steps:
    
    # 1. Add the first two numbers: 100 + 520
    # 2. Add the result from step 1 to the third number: (100 + 520) + 60
    
    # Step 1: Add 100 and 520
    # 100 + 520 = 620
    
    # Step 2: Add the result from step 1 to the third number (60)
    # (620) + 60 = 680
    
    # So, the sum of 100, 520, and 60 is 680.

    return response_assistant_only


#Run this code:
device = fetch_device()
hf_model = hf_model
system_input = "You are a math expert assistant. Your mission is to help users understand and solve various math problems. You should provide step-by-step solutions, explain reasonings and give the correct answer."
user_input = "calculate 100 + 520 + 60"


def generate_model_card_code_snippet(self):
    '''
    Coming soon.
    '''
    return 

def try_except_for_code_snippet(self):
    try:
        response = self.generate_response(
            model = 
        )
    except Exception as e:
        return 
        logging.error(traceback.format_exc())



    