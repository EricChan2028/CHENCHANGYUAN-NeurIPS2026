FROM python:3.11-slim
LABEL qfbench2.interface_version="2.0"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY vendor/track3_runtime /app/vendor/track3_runtime
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone https://github.com/jpmorganchase/abides-jpmc-public.git /tmp/abides \
    && git -C /tmp/abides checkout f9cbe51342b7dedd9587e4e069040d68a5c6477f \
    && git -C /tmp/abides apply /app/vendor/track3_runtime/patches/order_size_model.pomegranate-free.patch \
    && git -C /tmp/abides apply /app/vendor/track3_runtime/patches/kernel_message_ledger.patch \
    && git -C /tmp/abides apply /app/vendor/track3_runtime/patches/exchange_protocol_stp.patch \
    && git -C /tmp/abides apply /app/vendor/track3_runtime/patches/oracle_scheduled_jump.patch \
    && pip install --no-deps /tmp/abides/abides-core /tmp/abides/abides-markets \
    && apt-get purge -y git && apt-get autoremove -y \
    && rm -rf /tmp/abides /var/lib/apt/lists/*
COPY cpu_agent.py /app/cpu_agent.py
COPY LICENSE /app/LICENSE
ENTRYPOINT ["python", "/app/cpu_agent.py"]
CMD ["--help"]
