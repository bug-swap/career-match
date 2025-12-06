package com.careermatch.backend.dto.response;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ApiResponseTest {

    @Test
    void testBuilder() {
        ApiResponse<String> response = ApiResponse.<String>builder()
                .success(true)
                .message("OK")
                .data("test")
                .timestamp(123L)
                .build();

        assertTrue(response.getSuccess());
        assertEquals("OK", response.getMessage());
        assertEquals("test", response.getData());
        assertEquals(123L, response.getTimestamp());
    }

    @Test
    void testNoArgsAndSetters() {
        ApiResponse<String> response = new ApiResponse<>();
        response.setSuccess(false);
        response.setMessage("error");
        response.setData("data");
        response.setTimestamp(456L);

        assertFalse(response.getSuccess());
        assertEquals("error", response.getMessage());
        assertEquals("data", response.getData());
        assertEquals(456L, response.getTimestamp());
    }

    @Test
    void testAllArgsConstructor() {
        ApiResponse<Integer> response = new ApiResponse<>(true, "msg", 42, 789L);

        assertTrue(response.getSuccess());
        assertEquals("msg", response.getMessage());
        assertEquals(42, response.getData());
        assertEquals(789L, response.getTimestamp());
    }

    @Test
    void testSuccessFactory() {
        ApiResponse<String> response = ApiResponse.success("data");

        assertTrue(response.getSuccess());
        assertEquals("data", response.getData());
        assertNotNull(response.getTimestamp());
    }

    @Test
    void testErrorFactory() {
        ApiResponse<String> response = ApiResponse.error("error message");

        assertFalse(response.getSuccess());
        assertEquals("error message", response.getMessage());
        assertNotNull(response.getTimestamp());
    }
}
