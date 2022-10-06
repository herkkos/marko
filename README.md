# MARKO
## Mimicking Actual Real time Konversation Outputs

Marko is a chatbot framework designed to fulfil following requirements:
- Work with grammatical cases of Finnish language
- Interpret and generate Finnish texting language (similar to spoken language)
- Be able to be trained without a label set or necessary a hard coded word list

## Updates
### 2022-10-06
By using BERT tokens and transformer network the model succeeded to generate mostly credible texting language. 
Interpretation relies on good will and hope. Through testing with a group consisting of non-tech-
savvy people the model revealed to be convincing.

TODOs:
- Separate interpretation and generation from each other in the transformer implementation.
- Generate high-level labels to be used along BERT tokens. E.g. 'question', 'reaction', 'sarcastic'.
- Add randomness to training data
- Study clustering topics in order to create a information network. This could be used in generating responses
- Add restrictions to message replying to make the bot more human like.
