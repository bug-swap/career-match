package com.careermatch.backend.exception;

/**
 * Raised when the Python ML microservice responds with an error or cannot be
 * reached.
 */
public class MLServiceException extends RuntimeException {
    public MLServiceException(String message) {
        super(message);
    }

    public MLServiceException(String message, Throwable cause) {
        super(message, cause);
    }
}
