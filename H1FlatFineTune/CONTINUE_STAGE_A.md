# Stage A mở rộng: tiếp tục checkpoint 3.000

Ngày 10/9/2026: chỉ chuẩn bị code/lệnh, không chạy training.

- Stage A: `vx ∈ [-0.5, 2.0] m/s`; `vy=0`.
- 60% nhóm tiến được lấy mẫu trong `[1,2] m/s` (36% tổng lần lấy mẫu).
- Giữ 64 env, 5 PPO epochs, LR fixed `1e-4`, reward và giới hạn gia tốc/yaw.
- `--resume` giữ actor, critic, optimizer, std, iteration và RNG. Không gọi warm-start/reset std.
- Môi trường được khởi tạo lại; không phục hồi trạng thái vật lý/rollout buffer giữa episode.
- `--allow_command_range_change` ghi nhận đổi mục tiêu của cùng stage từ `[-0.3,1]` sang `[-0.5,2]`.
- `--max_iterations 6000` là tổng mốc stage: resume 3.000 sẽ chạy thêm 3.000, không phải thêm 6.000.
- Mặc định vẫn 3.000 nếu không truyền cap; vì vậy cần dùng đúng lệnh bên dưới.

## CMD — chạy lệnh này sẽ thực sự train

```bat
cd /d D:\IssacSim\H1FlatFineTune
D:\IssacSim\env_isaaclab\Scripts\python.exe -B scripts\train.py --stage A --resume "D:\IssacSim\H1FlatFineTune\logs\20260907T155849_808354Z_train\A_3000\checkpoint.pt" --allow_command_range_change --num_envs 64 --max_iterations 6000 --seed 42 --headless
```

Output tạo trong run mới dưới `logs`; checkpoint gốc không bị ghi đè. Lưu các mốc 3.500, 4.000, …, 6.000 và `checkpoint.pt` cuối run; train không reload scene ở các mốc lưu.

Checkpoint mới đã có `command_range` nên lần resume kế tiếp cùng range không cần flag xác nhận thay đổi.

## Evaluate và giới hạn còn lại

Stage A kiểm tra đứng, quay trái/phải, vận tốc `-0.5, -0.3, 0.5, 1.0, 1.5, 2.0 m/s` và chuỗi tiến–dừng–lùi. Không nới gate survival/MAE. Checkpoint 3.000 cũ chưa đạt quay trái, nên không tuyên bố range mới đã được kiểm định.

Chỉ thay range stage A. B-E chưa đổi; B có vmax 1.5 < A mới 2.0, vì vậy `--curriculum` bị chặn để tránh tự động chuyển sang range hẹp hơn. Roadmap nâng tốc độ tiếp theo cần điều chỉnh riêng.
