# Tham-Luan-Quantum

Mã đo, sổ tay Google Colab và dữ liệu thô cho tham luận **"Máy tính của sinh viên mô phỏng được bao nhiêu qubit?"** (Hội thảo "Định hướng phát triển công nghệ lượng tử tại TP.HCM", 27/11/2026).

Tác giả: Trương Ngô Phát, Trường Đại học Công Thương TP. Hồ Chí Minh.

## Nội dung
- `bench.py` – đo trên laptop (Qiskit Aer, PennyLane default.qubit / lightning.qubit, NumPy tự viết, CuPy tự viết).
- `colab_bench.py` – bản chạy được trên Linux/Colab (biến môi trường `SIMS`, `OUT`, `NMAX`).
- `notebooks/` – ba sổ tay Colab (CPU và GPU T4).
- `data/` – kết quả thô: `bench.json`, `extra.log` (laptop), `colab_run1/2/3.json` (Colab), `colab_gpu.json` (lượt GPU không có kết quả).
- `paper/` – bản thảo tham luận (.docx).
- `fig3.png` – hình thời gian theo số qubit.

## Phương pháp
Mỗi lần đo chạy trong tiến trình riêng, khởi động nóng ở n = 4, n ≤ 18 lấy trung vị 3 lần, n lớn đo 1 lần; giới hạn 240 giây; kiểm tra tính đúng đắn bằng ⟨Z₀⟩ giữa các bộ mô phỏng. Bộ nhớ vector trạng thái: 16·2ⁿ byte (complex128).

## Chạy lại
```bash
pip install numpy psutil qiskit qiskit-aer pennylane pennylane-lightning
python bench.py
```
