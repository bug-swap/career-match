package com.careermatch.backend.service;

import com.careermatch.backend.dto.response.JobWithScore;
import com.careermatch.backend.dto.response.SimilarJobsResponse;
import com.careermatch.backend.exception.MLServiceException;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class RpcServiceTest {

    private MockWebServer mockWebServer;
    private RpcService rpcService;

    @BeforeEach
    void setUp() throws IOException {
        mockWebServer = new MockWebServer();
        mockWebServer.start();

        WebClient webClient = WebClient.builder()
                .baseUrl(mockWebServer.url("/").toString())
                .build();

        rpcService = new RpcService(webClient);
        ReflectionTestUtils.setField(rpcService, "supabaseAnonKey", "test-anon-key");
    }

    @AfterEach
    void tearDown() throws IOException {
        mockWebServer.shutdown();
    }

    @Test
    @DisplayName("Should get similar jobs by category successfully")
    void getSimilarJobsByCategory_Success() {
        String responseJson = """
                [
                    {
                        "id": "1",
                        "title": "Software Engineer",
                        "company": "Tech Corp",
                        "location": "San Francisco, CA",
                        "job_type": "Full-time",
                        "is_remote": true,
                        "min_amount": 100000,
                        "max_amount": 150000,
                        "currency": "USD",
                        "job_url": "https://example.com/job/1",
                        "category": "ENGINEERING",
                        "score": 0.95
                    },
                    {
                        "id": "2",
                        "title": "Senior Developer",
                        "company": "Startup Inc",
                        "location": "New York, NY",
                        "job_type": "Full-time",
                        "is_remote": false,
                        "category": "ENGINEERING",
                        "score": 0.88
                    }
                ]
                """;

        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody(responseJson));

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        SimilarJobsResponse response = rpcService.getSimilarJobsByCategory(embedding, "ENGINEERING", 10);

        assertNotNull(response);
        assertTrue(response.getSuccess());
        assertEquals(2, response.getCount());
        assertNotNull(response.getJobs());
        assertEquals(2, response.getJobs().size());

        JobWithScore firstJob = response.getJobs().get(0);
        assertEquals("1", firstJob.getId());
        assertEquals("Software Engineer", firstJob.getTitle());
        assertEquals("Tech Corp", firstJob.getCompany());
        assertEquals(0.95, firstJob.getScore());
    }

    @Test
    @DisplayName("Should handle empty job list")
    void getSimilarJobsByCategory_EmptyList() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody("[]"));

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3 };
        SimilarJobsResponse response = rpcService.getSimilarJobsByCategory(embedding, "RARE_CATEGORY", 10);

        assertNotNull(response);
        assertTrue(response.getSuccess());
        assertEquals(0, response.getCount());
        assertNotNull(response.getJobs());
        assertTrue(response.getJobs().isEmpty());
    }

    @Test
    @DisplayName("Should use default limit when null")
    void getSimilarJobsByCategory_NullLimit() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody("[]"));

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3 };
        SimilarJobsResponse response = rpcService.getSimilarJobsByCategory(embedding, "ENGINEERING", null);

        assertNotNull(response);
        assertTrue(response.getSuccess());
    }

    @Test
    @DisplayName("Should throw MLServiceException on 4xx error")
    void getSimilarJobsByCategory_4xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(400)
                .setBody("Bad Request - Invalid category"));

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3 };
        assertThrows(MLServiceException.class,
                () -> rpcService.getSimilarJobsByCategory(embedding, "INVALID", 10));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 5xx error")
    void getSimilarJobsByCategory_5xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(500)
                .setBody("Internal Server Error"));

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3 };
        assertThrows(MLServiceException.class,
                () -> rpcService.getSimilarJobsByCategory(embedding, "ENGINEERING", 10));
    }

    @Test
    @DisplayName("Should throw MLServiceException on connection error")
    void getSimilarJobsByCategory_ConnectionError() throws IOException {
        mockWebServer.shutdown();

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3 };
        assertThrows(MLServiceException.class,
                () -> rpcService.getSimilarJobsByCategory(embedding, "ENGINEERING", 10));
    }

    @Test
    @DisplayName("Should include proper headers in request")
    void getSimilarJobsByCategory_IncludesHeaders() throws InterruptedException {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody("[]"));

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3 };
        rpcService.getSimilarJobsByCategory(embedding, "ENGINEERING", 10);

        var request = mockWebServer.takeRequest();
        assertEquals("test-anon-key", request.getHeader("apikey"));
        assertEquals("Bearer test-anon-key", request.getHeader("Authorization"));
        assertEquals("/rpc/get_similar_jobs_by_category", request.getPath());
    }
}
