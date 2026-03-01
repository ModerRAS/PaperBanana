# Build stage
FROM rust:1.83-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better caching
COPY Cargo.toml ./

# Create a dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs
RUN cargo build --release || true
RUN rm -rf src

# Copy the actual source code
COPY src ./src
COPY configs ./configs
COPY style_guides ./style_guides

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /app/target/release/paper_banana /app/paper_banana

# Copy necessary data files
COPY configs ./configs
COPY style_guides ./style_guides

# Ensure data directory exists
RUN mkdir -p data/PaperBananaBench/diagram data/PaperBananaBench/plot && chmod -R 777 data

# Expose port (for future web UI)
EXPOSE 8080

# Run the application
ENTRYPOINT ["/app/paper_banana"]
