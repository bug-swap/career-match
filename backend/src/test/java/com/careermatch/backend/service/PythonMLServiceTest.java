package com.careermatch.backend.service;

import com.careermatch.backend.dto.response.*;
import com.careermatch.backend.exception.MLServiceException;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class PythonMLServiceTest {

    private MockWebServer mockWebServer;
    private PythonMLService pythonMLService;
    private MultipartFile mockResumeFile;

    @BeforeEach
    void setUp() throws IOException {
        mockWebServer = new MockWebServer();
        mockWebServer.start();

        WebClient webClient = WebClient.builder()
                .baseUrl(mockWebServer.url("/").toString())
                .build();

        pythonMLService = new PythonMLService(webClient);

        mockResumeFile = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test resume content".getBytes());
    }

    @AfterEach
    void tearDown() throws IOException {
        mockWebServer.shutdown();
    }

    @Test
    @DisplayName("Should extract sections successfully")
    void extractSections_Success() {
        String responseJson = """
                {
                    "success": true,
                    "sections": {
                        "education": [{"degree": "BS", "major": "CS", "institution": "MIT"}],
                        "experience": [],
                        "project": [],
                        "publication": [],
                        "contact": {"name": "John Doe", "email": "john@example.com"}
                    },
                    "metadata": {"section_count": 2, "total_items": 2}
                }
                """;

        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody(responseJson));

        SectionsResponse response = pythonMLService.extractSections(mockResumeFile);

        assertNotNull(response);
        assertTrue(response.isSuccess());
        assertNotNull(response.getSections());
        assertEquals("John Doe", response.getSections().getContact().getName());
    }

    @Test
    @DisplayName("Should extract entities successfully")
    void extractEntities_Success() {
        String responseJson = """
                {
                    "success": true,
                    "entities": {
                        "skills": ["Java", "Python"],
                        "experience_years": 5
                    }
                }
                """;

        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody(responseJson));

        EntitiesResponse response = pythonMLService.extractEntities(mockResumeFile);

        assertNotNull(response);
        assertTrue(response.getSuccess());
    }

    @Test
    @DisplayName("Should classify category successfully")
    void classifyCategory_Success() {
        String responseJson = """
                {
                    "success": true,
                    "classification": {
                        "category": "ENGINEERING",
                        "confidence": 0.95,
                        "top_3": [
                            {"category": "ENGINEERING", "confidence": 0.95},
                            {"category": "IT", "confidence": 0.03},
                            {"category": "DATA_SCIENCE", "confidence": 0.02}
                        ]
                    }
                }
                """;

        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody(responseJson));

        CategoryResponse response = pythonMLService.classifyCategory(mockResumeFile);

        assertNotNull(response);
        assertTrue(response.getSuccess());
        assertNotNull(response.getClassification());
        assertEquals("ENGINEERING", response.getClassification().getCategory());
        assertEquals(0.95, response.getClassification().getConfidence());
    }

    @Test
    @DisplayName("Should generate embedding successfully")
    void generateEmbedding_Success() {
        String responseJson = """
                {
                    "success": true,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
                """;

        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .setBody(responseJson));

        EmbeddingResponse response = pythonMLService.generateEmbedding(mockResumeFile);

        assertNotNull(response);
        assertTrue(response.getSuccess());
        assertNotNull(response.getEmbedding());
        assertEquals(5, response.getEmbedding().length);
    }

    @Test
    @DisplayName("Should throw MLServiceException on 4xx error for extractSections")
    void extractSections_4xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(400)
                .setBody("Bad Request"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.extractSections(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 5xx error for extractSections")
    void extractSections_5xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(500)
                .setBody("Internal Server Error"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.extractSections(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 4xx error for extractEntities")
    void extractEntities_4xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(400)
                .setBody("Bad Request"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.extractEntities(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 5xx error for extractEntities")
    void extractEntities_5xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(500)
                .setBody("Internal Server Error"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.extractEntities(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 4xx error for classifyCategory")
    void classifyCategory_4xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(400)
                .setBody("Bad Request"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.classifyCategory(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 5xx error for classifyCategory")
    void classifyCategory_5xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(500)
                .setBody("Internal Server Error"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.classifyCategory(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 4xx error for generateEmbedding")
    void generateEmbedding_4xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(400)
                .setBody("Bad Request"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.generateEmbedding(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on 5xx error for generateEmbedding")
    void generateEmbedding_5xxError() {
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(500)
                .setBody("Internal Server Error"));

        assertThrows(MLServiceException.class,
                () -> pythonMLService.generateEmbedding(mockResumeFile));
    }

    @Test
    @DisplayName("Should throw MLServiceException on connection error")
    void extractSections_ConnectionError() throws IOException {
        mockWebServer.shutdown();

        assertThrows(MLServiceException.class,
                () -> pythonMLService.extractSections(mockResumeFile));
    }
}
