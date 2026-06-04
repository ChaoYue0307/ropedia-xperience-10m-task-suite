FROM nginx:1.27-alpine

LABEL org.opencontainers.image.title="Ropedia Xperience-10M Task Suite"
LABEL org.opencontainers.image.description="Static research dashboard for the Ropedia Xperience-10M task-suite project."
LABEL org.opencontainers.image.source="https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite"
LABEL org.opencontainers.image.url="https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/"
LABEL org.opencontainers.image.licenses="MIT"

COPY docs/ /usr/share/nginx/html/
COPY README.md /usr/share/nginx/html/PROJECT_README.md
COPY PROJECT_STATUS.md /usr/share/nginx/html/PROJECT_STATUS.md
COPY ARTIFACT_GUIDE.md /usr/share/nginx/html/ARTIFACT_GUIDE.md
COPY FOUNDATION_MODEL_PLAN.md /usr/share/nginx/html/FOUNDATION_MODEL_PLAN.md
COPY RESEARCH_ROADMAP.md /usr/share/nginx/html/RESEARCH_ROADMAP.md
COPY XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md /usr/share/nginx/html/XPERIENCE_EMBODIED_FOUNDATION_MODEL_PRETRAINING.md
COPY DATA_NOTICE.md /usr/share/nginx/html/DATA_NOTICE.md
COPY LICENSE /usr/share/nginx/html/LICENSE
COPY CITATION.cff /usr/share/nginx/html/CITATION.cff

EXPOSE 80
