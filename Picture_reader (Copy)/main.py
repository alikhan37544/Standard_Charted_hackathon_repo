import ollama

response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['/home/shree-xd/Documents/llamaVision/Screenshot from 2025-03-17 22-07-59.png']
    }]
)

print(response)

'''The image shows a golden retriever dog with its tongue out, appearing to be panting. 
It's a color photo, and the dog has a happy expression. 
There seems to be a watermark or a logo overlayed on the image,
 indicating it may have been taken from a website or is part of a screen capture.'''