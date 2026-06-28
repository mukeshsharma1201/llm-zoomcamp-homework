.PHONY: dev

up:
	docker run -d \
    --name pgvector \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=faq \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17

down:
	docker rm -f pgvector

db-clear: down
	docker volume rm pgvector_data

db-logs:
	docker logs -f pgvector
