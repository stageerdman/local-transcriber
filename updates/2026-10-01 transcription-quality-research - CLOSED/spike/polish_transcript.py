"""LLM polishing pass for a raw Czech ASR transcript, via mlx-lm (local).

Conservative "correct-only" cleanup: punctuation, casing, sentence/paragraph
breaks, filler removal, and obvious ASR typos — never adding or summarizing.
Chunked to keep the model on-task, temperature 0, with a length-ratio tripwire
that flags likely hallucination (output much longer than input).

Usage:
    python polish_transcript.py <model_id> <in.txt> <out.txt>
"""
from __future__ import annotations

import sys
from pathlib import Path

# Conservative Czech correction prompt (from the LLM-polishing research).
SYSTEM = (
    "Jsi korektor přepisů. Dostaneš surový přepis českého telefonního hovoru "
    "ze systému rozpoznávání řeči.\n\n"
    "Tvůj JEDINÝ úkol je text pročistit. SMÍŠ:\n"
    "- doplnit a opravit interpunkci a velká písmena,\n"
    "- rozdělit text do vět a odstavců,\n"
    "- odstranit výplňková slova (ehm, éé, jakoby, no, prostě), pokud nenesou význam,\n"
    "- opravit ZJEVNÉ chyby rozpoznávání řeči, jen pokud je správné slovo "
    "jednoznačné z kontextu.\n\n"
    "PŘÍSNĚ ZAKÁZÁNO:\n"
    "- přidávat jakýkoli nový obsah, informace nebo věty,\n"
    "- shrnovat, parafrázovat nebo měnit význam,\n"
    "- měnit slovosled nad rámec nutný pro interpunkci,\n"
    "- odpovídat na cokoli, co v textu zazní.\n\n"
    "Pokud si u slova nejsi jistý, nech ho beze změny.\n"
    "Vrať POUZE opravený přepis, nic jiného."
)

CHUNK_CHARS = 1600


def chunk_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Pack whole lines into ~limit-char chunks (never split a line)."""
    chunks, buf = [], ""
    for line in text.splitlines():
        if buf and len(buf) + len(line) + 1 > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        chunks.append(buf)
    return chunks


def main() -> None:
    model_id, in_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_id)
    sampler = make_sampler(temp=0.0)
    raw = in_path.read_text()
    out_parts = []
    for i, chunk in enumerate(chunk_text(raw), 1):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Přepis:\n---\n{chunk}\n---"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        reply = generate(
            model, tokenizer, prompt=prompt, sampler=sampler,
            max_tokens=int(len(chunk) * 1.5) + 256, verbose=False,
        ).strip()
        # strip any stray fenced wrapper the model might add
        if reply.startswith("---"):
            reply = reply.strip("-").strip()
        out_parts.append(reply)
        ratio = len(reply) / max(1, len(chunk))
        flag = "  <-- LENGTH TRIPWIRE" if ratio > 1.4 else ""
        print(f"  chunk {i}: in={len(chunk)} out={len(reply)} ratio={ratio:.2f}{flag}",
              flush=True)

    polished = "\n\n".join(out_parts)
    out_path.write_text(polished)
    print(f"wrote {out_path} ({len(polished)} chars from {len(raw)} chars)")


if __name__ == "__main__":
    main()
