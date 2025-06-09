FROM alpine

COPY movement-classification /model

CMD ["ls", "/model"]
