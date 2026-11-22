from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
attention = model.transformer.h[0].attn

print("Number of heads:", attention.num_heads)
print("Head dimension:", attention.head_dim)

print("\nQKV projection:")
print("Weight:", attention.c_attn.weight.shape)
print("Bias:", attention.c_attn.bias.shape)

print("\nOutput projection:")
print("Weight:", attention.c_proj.weight.shape)
print("Bias:", attention.c_proj.bias.shape)