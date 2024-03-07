# LSTM

基于 numpy 实现的 LSTM 算法。

## 梯度检查

这里使用**梯度检查**来验证 LSTM 的反向传播的梯度是否正确，下面是运行的结果。

```text
parameter (0,0,0) expect_grad: -1.8619 backward_grad: -1.8619 [OK]
parameter (0,0,1) expect_grad: -0.4222 backward_grad: -0.4222 [OK]
parameter (0,0,2) expect_grad: -0.4999 backward_grad: -0.4999 [OK]
parameter (0,0,3) expect_grad: -0.3638 backward_grad: -0.3638 [OK]
parameter (0,0,4) expect_grad: 0.4320 backward_grad: 0.4320 [OK]
parameter (0,0,5) expect_grad: -0.6982 backward_grad: -0.6982 [OK]
parameter (0,0,6) expect_grad: -2.0696 backward_grad: -2.0696 [OK]
parameter (0,1,0) expect_grad: 0.1023 backward_grad: 0.1023 [OK]
parameter (0,1,1) expect_grad: -0.3802 backward_grad: -0.3802 [OK]
parameter (0,1,2) expect_grad: -0.0487 backward_grad: -0.0487 [OK]
parameter (0,1,3) expect_grad: 0.0044 backward_grad: 0.0044 [OK]
parameter (0,1,4) expect_grad: 0.0863 backward_grad: 0.0863 [OK]
parameter (0,1,5) expect_grad: 0.0790 backward_grad: 0.0790 [OK]
parameter (0,1,6) expect_grad: 0.7094 backward_grad: 0.7094 [OK]
parameter (0,2,0) expect_grad: -0.7426 backward_grad: -0.7426 [OK]
parameter (0,2,1) expect_grad: -0.6647 backward_grad: -0.6647 [OK]
parameter (0,2,2) expect_grad: -0.2931 backward_grad: -0.2931 [OK]
parameter (0,2,3) expect_grad: -0.1643 backward_grad: -0.1643 [OK]
parameter (0,2,4) expect_grad: 0.3077 backward_grad: 0.3077 [OK]
parameter (0,2,5) expect_grad: -0.2285 backward_grad: -0.2285 [OK]
parameter (0,2,6) expect_grad: -0.0925 backward_grad: -0.0925 [OK]
parameter (0,3,0) expect_grad: 0.1463 backward_grad: 0.1463 [OK]
parameter (0,3,1) expect_grad: 0.4394 backward_grad: 0.4394 [OK]
parameter (0,3,2) expect_grad: 0.1160 backward_grad: 0.1160 [OK]
parameter (0,3,3) expect_grad: 0.0443 backward_grad: 0.0443 [OK]
parameter (0,3,4) expect_grad: -0.1447 backward_grad: -0.1447 [OK]
parameter (0,3,5) expect_grad: 0.0139 backward_grad: 0.0139 [OK]
parameter (0,3,6) expect_grad: -0.4373 backward_grad: -0.4373 [OK]
parameter (0,4,0) expect_grad: 1.5111 backward_grad: 1.5111 [OK]
parameter (0,4,1) expect_grad: 0.7081 backward_grad: 0.7081 [OK]
parameter (0,4,2) expect_grad: 0.4748 backward_grad: 0.4748 [OK]
parameter (0,4,3) expect_grad: 0.3094 backward_grad: 0.3094 [OK]
parameter (0,4,4) expect_grad: -0.4503 backward_grad: -0.4503 [OK]
parameter (0,4,5) expect_grad: 0.5298 backward_grad: 0.5298 [OK]
parameter (0,4,6) expect_grad: 1.1400 backward_grad: 1.1400 [OK]
```