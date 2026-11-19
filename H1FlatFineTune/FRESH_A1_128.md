# Fresh pretrained fine-tune: 128 env, A1-A5 -> B-E

Ngày 12/9/2026. Chuẩn bị code/lệnh; không tự chạy training hoặc simulator.

Khởi tạo từ `checkpoints/base/checkpoint.pt` (published H1 Flat), không dùng checkpoint fine-tune 3000/6000.
Nạp actor/critic pretrained; std reset về 0.3, Adam mới, iteration bắt đầu từ 0.
Đây là fine-tune từ pretrained, không phải mạng ngẫu nhiên.

| Stage | vx (m/s) |
|---|---|
| A1 | [-0.30, 1.25] |
| A2 | [-0.35, 1.5] |
| A3 | [-0.40, 1.5] |
| A4 | [-0.45, 1.75] |
| A5 | [-0.50, 2.0] |
| B | [-0.625, 2.25] |
| C | [-0.750, 2.50] |
| D | [-0.875, 2.75] |
| E | [-1.000, 3.00] |

Giới hạn lùi giảm 0.05 m/s mỗi stage; evaluation giữ cả các mốc lùi đã đạt trước đó.

Từ A5 tới E (cập nhật 15/9/2026), mỗi stage tăng vmax 0.25 m/s và giảm vmin 0.125 m/s. B-E giữ lại các bài kiểm tra A1-A5. Chỉ mới sửa code, chưa kiểm định khả năng chạy ở range B-E mới.

Giữ 5 PPO epochs, LR fixed 1e-4, entropy coefficient 0.005, rollout 24 và 4 minibatches.
Không chỉnh reward, actuator hoặc observation/action contract.
128 env tạo 3072 transitions/iteration; 3000 iterations tạo 9,216,000 transitions.
VRAM cho 128 env chưa được smoke test trên máy trong lần sửa này.

## CMD: chỉ A1, iteration 0 tới 3000

Lệnh bên dưới thực sự khởi chạy training khi bạn chạy:

```bat
cd /d D:\IssacSim\H1FlatFineTune
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage A1 --init_checkpoint "D:\IssacSim\H1FlatFineTune\checkpoints\base\checkpoint.pt" --num_envs 128 --max_iterations 3000 --seed 42 --headless
```

Không dùng --resume hoặc --allow_command_range_change cho lượt khởi tạo này.
Output: logs/<timestamp>_train/A1_3000; không ghi đè run/checkpoint cũ.
Checkpoint mỗi 500 iterations: checkpoint_0500.pt ... checkpoint_3000.pt, cùng checkpoint.pt cuối lượt.
Không evaluate hoặc reload scene ở các mốc lưu. Evaluate ở cuối 3000 iterations.
--stage A1 dừng sau đánh giá, không tự train A2.

## Sau khi A1 đạt: chuyển thủ công sang A2

Thay PATH bằng folder A1_3000 của run mới. Phải dùng report đã đạt của đúng checkpoint.

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage A2 --resume "PATH\checkpoint.pt" --promotion_report "PATH\evaluation\evaluation.json" --num_envs 128 --max_iterations 3000 --seed 42 --headless
```

Stage A2 có stage_iterations bắt đầu từ 0, nhưng completed_iterations tích lũy từ A1.
Giữ optimizer/std đã học khi chuyển stage. Chuyển A3-A5 tương tự, không bỏ qua stage.
Nếu fail, không tự sang stage tiếp theo. Không thay gate để làm report đạt.

## Curriculum tự động (tùy chọn, dài hơn lệnh A1)

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --curriculum --init_checkpoint "D:\IssacSim\H1FlatFineTune\checkpoints\base\checkpoint.pt" --num_envs 128 --max_iterations 3000 --seed 42 --headless
```

3000 iterations mỗi stage, tối đa 27000 tổng cộng nếu cả chín stage A1-A5 và B-E đều đạt.
Chỉ chuyển stage khi evaluation đầy đủ đạt; fail thì dừng.
Lệnh này không phải chỉ chạy tổng cộng 3000 iterations.

Evaluation mỗi stage kiểm tra đứng, quay hai hướng, chuỗi tiến-dừng-lùi, mốc hiện tại và mốc đã đạt.
A1 không bị bắt evaluate tốc độ 1.5/2.0 ngoài range. A5 giữ các mốc 1.25/1.5/1.75 đã học.
Gate survival/MAE giữ nguyên. Không tuyên bố đạt tốc độ khi chưa có report.

Lộ trình chính: A1 -> A2 -> A3 -> A4 -> A5 -> B -> C -> D -> E. E là bước cuối.
Stage A giữ làm entry point cũ. B-E đã đổi range; không sửa checkpoint/log/report cũ.
Resume cùng stage B-E từ range cũ cần xác nhận --allow_command_range_change; report cũ không dùng để promote range mới.
Chuyển A5 -> B dùng --resume checkpoint A5 và --promotion_report report A5 đã pass, không dùng --init_checkpoint.
Với --stage B --max_iterations 3000, chỉ train B thêm 3000 iterations rồi evaluate và dừng; giữ optimizer/std, stage_iterations về 0 nhưng completed_iterations tiếp tục tích lũy.
