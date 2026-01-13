run:
    ./.venv/bin/uvicorn server:app --uds /dev/shm/generator.sock
