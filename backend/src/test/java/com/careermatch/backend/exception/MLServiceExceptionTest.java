package com.careermatch.backend.exception;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MLServiceExceptionTest {

    @Test
    @DisplayName("Should create exception with message")
    void constructor_WithMessage() {
        String message = "Test error message";
        MLServiceException exception = new MLServiceException(message);

        assertEquals(message, exception.getMessage());
        assertNull(exception.getCause());
    }

    @Test
    @DisplayName("Should create exception with message and cause")
    void constructor_WithMessageAndCause() {
        String message = "Test error message";
        Throwable cause = new RuntimeException("Root cause");

        MLServiceException exception = new MLServiceException(message, cause);

        assertEquals(message, exception.getMessage());
        assertEquals(cause, exception.getCause());
    }

    @Test
    @DisplayName("Should handle null message")
    void constructor_NullMessage() {
        MLServiceException exception = new MLServiceException(null);

        assertNull(exception.getMessage());
    }

    @Test
    @DisplayName("Should handle null cause")
    void constructor_NullCause() {
        MLServiceException exception = new MLServiceException("Message", null);

        assertEquals("Message", exception.getMessage());
        assertNull(exception.getCause());
    }

    @Test
    @DisplayName("Should be instance of RuntimeException")
    void isRuntimeException() {
        MLServiceException exception = new MLServiceException("Test");

        assertTrue(exception instanceof RuntimeException);
    }

    @Test
    @DisplayName("Should be throwable")
    void isThrowable() {
        MLServiceException exception = new MLServiceException("Test");

        assertThrows(MLServiceException.class, () -> {
            throw exception;
        });
    }

    @Test
    @DisplayName("Should preserve stack trace")
    void preservesStackTrace() {
        MLServiceException exception = new MLServiceException("Test");

        assertNotNull(exception.getStackTrace());
        assertTrue(exception.getStackTrace().length > 0);
    }

    @Test
    @DisplayName("Should chain exceptions correctly")
    void chainingExceptions() {
        Exception rootCause = new IllegalArgumentException("Root");
        MLServiceException middleCause = new MLServiceException("Middle", rootCause);
        MLServiceException topException = new MLServiceException("Top", middleCause);

        assertEquals("Top", topException.getMessage());
        assertEquals(middleCause, topException.getCause());
        assertEquals(rootCause, topException.getCause().getCause());
    }
}
