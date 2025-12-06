package com.careermatch.backend.config;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.reactive.function.client.WebClient;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class WebClientConfigTest {

    @Test
    @DisplayName("Should create pythonMLWebClient bean")
    void pythonMLWebClient() {
        WebClientConfig config = new WebClientConfig();
        ReflectionTestUtils.setField(config, "pythonServiceUrl", "http://localhost:8000");
        ReflectionTestUtils.setField(config, "rpcServiceUrl", "http://localhost:8545");

        WebClient webClient = config.pythonMLWebClient();

        assertNotNull(webClient);
    }

    @Test
    @DisplayName("Should create rpcWebClient bean")
    void rpcWebClient() {
        WebClientConfig config = new WebClientConfig();
        ReflectionTestUtils.setField(config, "pythonServiceUrl", "http://localhost:8000");
        ReflectionTestUtils.setField(config, "rpcServiceUrl", "http://localhost:8545");

        WebClient webClient = config.rpcWebClient();

        assertNotNull(webClient);
    }

    @Test
    @DisplayName("Should use custom python service URL")
    void customPythonServiceUrl() {
        WebClientConfig config = new WebClientConfig();
        ReflectionTestUtils.setField(config, "pythonServiceUrl", "http://custom-ml-service:9000");
        ReflectionTestUtils.setField(config, "rpcServiceUrl", "http://localhost:8545");

        WebClient webClient = config.pythonMLWebClient();

        assertNotNull(webClient);
    }

    @Test
    @DisplayName("Should use custom RPC service URL")
    void customRpcServiceUrl() {
        WebClientConfig config = new WebClientConfig();
        ReflectionTestUtils.setField(config, "pythonServiceUrl", "http://localhost:8000");
        ReflectionTestUtils.setField(config, "rpcServiceUrl", "https://supabase-project.supabase.co/rest/v1");

        WebClient webClient = config.rpcWebClient();

        assertNotNull(webClient);
    }
}
