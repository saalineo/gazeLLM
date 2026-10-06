use gazellm::net::grpc::gaze_v1::inference_service_client::InferenceServiceClient;
use gazellm::net::grpc::GenerateRequest;
use gazellm::net::grpc_stream::start_grpc_server_with_shutdown;
use tokio_stream::StreamExt;

#[tokio::test]
async fn test_live_grpc_streaming_over_network() {
    // Arrange: bind ephemeral TCP listener and launch gRPC server with shutdown hook
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("ephemeral port should bind");
    let addr = listener.local_addr().expect("valid socket address");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let server_task = tokio::spawn(async move {
        start_grpc_server_with_shutdown(addr, async {
            let _ = shutdown_rx.await;
        })
        .await
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let target_uri = format!("http://{addr}");
    let mut client = InferenceServiceClient::connect(target_uri)
        .await
        .expect("client should connect to local gRPC server");

    let request = GenerateRequest {
        request_id: "net-test-stream".to_string(),
        prompt: "hello from live client".to_string(),
        max_new_tokens: 5,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
    };

    // Act: invoke streaming RPC and collect all token chunks
    let response = client
        .stream_generate(request)
        .await
        .expect("stream_generate RPC call should succeed");

    let mut stream = response.into_inner();
    let mut received_tokens = Vec::new();

    while let Some(item) = stream.next().await {
        let resp = item.expect("valid token packet");
        received_tokens.push(resp);
    }

    // Assert: verify token sequencing, identifiers, terminal markers, and clean shutdown
    assert_eq!(received_tokens.len(), 5);
    for (i, token) in received_tokens.iter().enumerate() {
        let expected_token_id = i32::try_from(i).expect("index within i32 range");
        assert_eq!(token.request_id, "net-test-stream");
        assert_eq!(token.token_id, expected_token_id);
        assert_eq!(token.token_text, format!("token_{i} "));
        if i == 4 {
            assert!(token.is_finished);
            assert_eq!(token.finish_reason, "stop");
        } else {
            assert!(!token.is_finished);
            assert_eq!(token.finish_reason, "");
        }
    }

    let _ = shutdown_tx.send(());
    let server_exit_status = server_task.await.expect("server task join");
    assert!(server_exit_status.is_ok());
}

#[tokio::test]
async fn test_live_grpc_client_disconnect_cancellation() {
    // Arrange: start gRPC server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("ephemeral port should bind");
    let addr = listener.local_addr().expect("valid socket address");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let server_task = tokio::spawn(async move {
        start_grpc_server_with_shutdown(addr, async {
            let _ = shutdown_rx.await;
        })
        .await
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let target_uri = format!("http://{addr}");
    let mut client = InferenceServiceClient::connect(target_uri)
        .await
        .expect("client should connect");

    let request = GenerateRequest {
        request_id: "disconnect-test".to_string(),
        prompt: "test disconnect".to_string(),
        max_new_tokens: 10,
        ..Default::default()
    };

    // Act: request stream, receive first chunk, then drop stream immediately
    let response = client
        .stream_generate(request)
        .await
        .expect("stream_generate should succeed");

    let mut stream = response.into_inner();
    let first_token = stream.next().await;

    // Assert: first token arrived, and dropping stream cancels gracefully without hanging server
    assert!(first_token.is_some());
    drop(stream);

    let _ = shutdown_tx.send(());
    let server_exit_status = server_task.await.expect("server task join");
    assert!(server_exit_status.is_ok());
}
