# Admin Client UI (React + Tailwind) served by Nginx
ARG NODE_VERSION=18

FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# Build and normalize output to /out (supports Vite "dist" or CRA "build")
RUN npm run build && mkdir -p /out && \
    if [ -d dist ]; then cp -r dist/* /out/; elif [ -d build ]; then cp -r build/* /out/; else echo "No dist or build output found"; exit 1; fi

FROM nginx:1.25-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /out/ /usr/share/nginx/html/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Browser base URL for Admin backend on host
ENV API_BASE_URL=http://localhost:5002

EXPOSE 80
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]