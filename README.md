# Speculative Decoding

This project focuses on optimizing speculative decoding for faster language model inference. 
The method uses a smaller, faster draft model to generate candidate tokens, which are then verified by a larger target model in a single forward pass.
