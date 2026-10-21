# MiMo-V2.6-Pro on an RTX 4090 (Windows)

The engine, its CUDA DLL, `coli` and the launcher measured on one machine: Ryzen 9
7950X3D, 192 GB DDR5, RTX 4090, two NVMe volumes. Every greedy reply is the byte-for-
byte reply of the colibri 2.0 engine; nothing is quantized beyond the checkpoint.
The history and the measurements: [README](https://github.com/CyberRSR/colibri#this-fork-mimo-v26-pro-on-one-rtx-4090),
[docs/mimo-cuda.md](https://github.com/CyberRSR/colibri/blob/main/docs/mimo-cuda.md).

## Быстрый старт (по-русски)

1. Нужны: Windows 10/11 x64, процессор с AVX-512 (Ryzen 7000/9000 или новее),
   NVIDIA RTX 40xx с драйвером для CUDA 12 или 13, Python 3.10+ в PATH, около 192 ГБ
   ОЗУ и модель `XiaomiMiMo/MiMo-V2.6-Pro` (535 ГиБ) на быстрых NVMe.
2. Распакуйте архив, например, в `C:\colibri`.
3. Запустите `START-MIMO.cmd` (по умолчанию модель в `D:\AI\Models\MiMo-V2.6-Pro-MOPD`,
   часть шардов в `C:\AI\MiMoPro-Shards`). Свои пути:
   `START-MIMO.cmd -ModelDir E:\MiMo-Pro -ShardDir ""` (пустая строка — все шарды в папке модели).
4. Через ~20 с API готов: `http://127.0.0.1:18080/v1`, модель `mimo-v2.6-pro-mopd`
   (OpenAI `/v1/chat/completions` и Anthropic `/v1/messages`). Контекст — 1 048 576
   токенов, ответ — до 65 536 токенов (рассуждение входит в него).

Для агентов (MiMo Code, DeepSeek-харнессы и т. п.): включайте потоковый вывод
(stream) — шлюз шлёт keep-alive, пока обрабатывается промпт; ведите один разговор за
раз — следующий ход переиспользует уже обработанный префикс, а посторонний запрос с
другим системным промптом его сбрасывает. Префикс переиспользуется и тогда, когда агент
присылает ответы модели обратно без рассуждений (`reasoning_content`), как делает
большинство OpenAI-совместимых клиентов: движок делает «снимок» конца каждого промпта, и
следующий ход продолжает с него (при 26 700 токенах: 7–34 с вместо 97–100 с в r4,
ответы байт-в-байт те же). Длинный промпт обрабатывается на GPU:
первый промпт в 25 000 токенов — около полутора минут (в r2 — около 4 минут, на CPU —
около часа), дальше — только новые токены. Ответ после такого промпта генерируется со
скоростью около 2,3 токена в секунду (в r3 — 1,4): внимание шага декодирования считается
на всех ядрах, байт-в-байт как раньше. Пока идёт промпт, GPU загружен почти
полностью, а видеопамять нужна движку: тяжёлые графические программы рядом могут его
замедлить.

## Requirements

- Windows 10/11 x64; a CPU with AVX-512 (`mimo.exe` is built for Zen 4: Ryzen
  7000/9000, or a recent Intel with AVX-512). Other CPUs: build `mimo.exe` with
  `ARCH=x86-64-v3` (docs/mimo-cuda.md).
- An NVIDIA RTX 40xx (the DLL is built for sm_89) and a driver for CUDA 12 or 13; the
  CUDA runtime is linked into `coli_cuda_mimo.dll`, nothing else to install.
- Python 3.10 or newer on PATH (`coli` is a Python script).
- About 192 GB of RAM for this profile (`-Cap 104` experts a layer; smaller machines:
  a lower `-Cap`, about 1.4 GB less per step of 1), the checkpoint on NVMe drives.

## Run

```bat
START-MIMO.cmd
START-MIMO.cmd -ModelDir E:\MiMo-V2.6-Pro -ShardDir ""
```

| Parameter | Default | Meaning |
|---|---|---|
| `-ModelDir` | `D:\AI\Models\MiMo-V2.6-Pro-MOPD` | The checkpoint (config.json, tokenizer, shards) |
| `-ShardDir` | `C:\AI\MiMoPro-Shards` | More shards on another drive (both drives read at once); `""`: none |
| `-Context` | 1048576 | Context; the attention caches take memory only as a conversation grows (102 KB a token) |
| `-MaxOutput` | 65536 | Most tokens a reply may have, reasoning included |
| `-Port` | 18080 | Port on 127.0.0.1 |
| `-Cap` | 104 | Experts cached per MoE layer (one pool for all layers) |
| `-Threads` | 16 | CPU workers (the physical cores) |
| `-DenseGB` / `-ReserveGB` | 20 / 2 | VRAM for the dense matrices / left free |
| `-OprojInt8` | off | Faster (+6%) and **not exact**: every o_proj on the GPU as int8 |

## Speed on the measured machine

| | colibri 2.0 | this release |
|---|---:|---:|
| benchmark request (64 tokens, greedy), tok/s | 1.159 | 2.67 |
| decode, tok/s | 1.359 | 3.32 |
| eight different prompts, tok/s | 1.061 | 2.26 |
| first token of a 4310-token prompt | -- | 47 s (r2: 69 s; 414 s with prompts on the CPU) |
| first token of a 24,957-token prompt | -- | 92 s (r2: 245 s; about an hour on the CPU) |
| a reply after that prompt (128 tokens, greedy), tok/s | -- | 2.32 (r3: 1.40) |
| a reply after the 4310-token prompt, tok/s | -- | 2.79 (r3: 2.62) |
| the next agent turn at 26,700 tokens, the client dropping the reasoning: first token | -- | 7-34 s (r4: 97-100 s) |

`tools\benchmark_mimo_cuda.py` measures your own machine (docs/mimo-cuda.md).
