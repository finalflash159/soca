# Attention trong Transformer

Attention cho phép mỗi token chọn ngữ cảnh liên quan từ các token khác rồi trộn thông tin đó vào biểu diễn của mình. 

- Query hỏi token hiện tại cần thông tin gì.
- Key biểu diễn pattern mà mỗi token có.
- Value chứa nội dung được lấy về.
- Điểm attention so sánh Query với Key, sau đó softmax tạo trọng số để trộn các Value. 

| Thành phần | Vai trò |
|---|---|
| Query và Key | Tính mức độ liên quan |
| Value | Nội dung được tổng hợp |

Công thức là:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$ 

Phép chia cho $\sqrt{d}$ giúp giữ điểm số trong vùng dễ học hơn, tránh softmax quá nhọn và làm gradient nhỏ. 

```python
scores = (Q @ K.T) / (d ** 0.5)
weights = softmax(scores)
output = weights @ V
```

Trong self-attention, Q, K và V đến từ cùng một chuỗi; causal self-attention chặn token tương lai.