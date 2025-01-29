mod models;
mod routes;
mod video_processor;

use axum::Router;
use routes::create_routes;
use tower::ServiceBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .init();

    let app = create_routes().layer(
        ServiceBuilder::new()
            .trace_for_http()
    );

    let addr = "0.0.0.0:3000".parse().unwrap();
    tracing::info!("Listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
