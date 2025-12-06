package com.careermatch.backend.parse;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ResumeParserFactoryTest {

    @Mock
    private ParsingStrategyFactory parsingStrategyFactory;

    @Mock
    private ResumeParsingStrategy mockStrategy;

    @InjectMocks
    private ResumeParserFactory resumeParserFactory;

    private MultipartFile mockPdfFile;
    private Resume mockResume;

    @BeforeEach
    void setUp() {
        mockPdfFile = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test PDF content".getBytes());

        mockResume = Resume.builder()
                .success(true)
                .sections(Resume.ResumeSection.builder()
                        .contact(Resume.Contact.builder()
                                .name("John Doe")
                                .build())
                        .education(Collections.emptyList())
                        .experience(Collections.emptyList())
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .build())
                .build();
    }

    @Test
    @DisplayName("Should parse with AI strategy")
    void parse_WithAIStrategy() throws Exception {
        when(parsingStrategyFactory.getStrategy(ParsingStrategyType.AI)).thenReturn(mockStrategy);
        when(mockStrategy.parse(any(MultipartFile.class))).thenReturn(mockResume);

        Resume result = resumeParserFactory.parse(mockPdfFile, ParsingStrategyType.AI);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        verify(parsingStrategyFactory).getStrategy(ParsingStrategyType.AI);
        verify(mockStrategy).parse(mockPdfFile);
    }

    @Test
    @DisplayName("Should parse with ML strategy")
    void parse_WithMLStrategy() throws Exception {
        when(parsingStrategyFactory.getStrategy(ParsingStrategyType.ML)).thenReturn(mockStrategy);
        when(mockStrategy.parse(any(MultipartFile.class))).thenReturn(mockResume);

        Resume result = resumeParserFactory.parse(mockPdfFile, ParsingStrategyType.ML);

        assertNotNull(result);
        verify(parsingStrategyFactory).getStrategy(ParsingStrategyType.ML);
    }

    @Test
    @DisplayName("Should parse with default strategy when null")
    void parse_WithNullStrategy() throws Exception {
        when(parsingStrategyFactory.getStrategy((ParsingStrategyType) null)).thenReturn(mockStrategy);
        when(mockStrategy.parse(any(MultipartFile.class))).thenReturn(mockResume);

        Resume result = resumeParserFactory.parse(mockPdfFile, null);

        assertNotNull(result);
        verify(parsingStrategyFactory).getStrategy((ParsingStrategyType) null);
    }

    @Test
    @DisplayName("Should parse using single argument method (default strategy)")
    void parse_SingleArgument() throws Exception {
        when(parsingStrategyFactory.getStrategy((ParsingStrategyType) null)).thenReturn(mockStrategy);
        when(mockStrategy.parse(any(MultipartFile.class))).thenReturn(mockResume);

        Resume result = resumeParserFactory.parse(mockPdfFile);

        assertNotNull(result);
        verify(parsingStrategyFactory).getStrategy((ParsingStrategyType) null);
    }

    @Test
    @DisplayName("Should throw exception when strategy parsing fails")
    void parse_StrategyThrowsException() throws Exception {
        when(parsingStrategyFactory.getStrategy(any(ParsingStrategyType.class))).thenReturn(mockStrategy);
        when(mockStrategy.parse(any(MultipartFile.class))).thenThrow(new RuntimeException("Parsing error"));

        RuntimeException exception = assertThrows(RuntimeException.class,
                () -> resumeParserFactory.parse(mockPdfFile, ParsingStrategyType.AI));

        assertTrue(exception.getMessage().contains("Failed to parse resume"));
    }

    @Test
    @DisplayName("Should extract text from TXT file")
    void extractTextFromFile_Txt() throws IOException {
        String content = "This is a test resume content";
        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                content.getBytes());

        String result = ResumeParserFactory.extractTextFromFile(txtFile);

        assertEquals(content, result);
    }

    @Test
    @DisplayName("Should throw exception for null filename")
    void extractTextFromFile_NullFilename() {
        MockMultipartFile fileWithNullName = new MockMultipartFile(
                "file",
                null,
                "application/octet-stream",
                "content".getBytes());

        assertThrows(IllegalArgumentException.class,
                () -> ResumeParserFactory.extractTextFromFile(fileWithNullName));
    }

    @Test
    @DisplayName("Should throw exception for file without extension")
    void extractTextFromFile_NoExtension() {
        MockMultipartFile fileNoExtension = new MockMultipartFile(
                "file",
                "resume",
                "application/octet-stream",
                "content".getBytes());

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> ResumeParserFactory.extractTextFromFile(fileNoExtension));

        assertTrue(exception.getMessage().contains("no extension"));
    }

    @Test
    @DisplayName("Should throw exception for unsupported format")
    void extractTextFromFile_UnsupportedFormat() {
        MockMultipartFile unsupportedFile = new MockMultipartFile(
                "file",
                "resume.xyz",
                "application/octet-stream",
                "content".getBytes());

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> ResumeParserFactory.extractTextFromFile(unsupportedFile));

        assertTrue(exception.getMessage().contains("Unsupported file format"));
    }

    @Test
    @DisplayName("Should throw exception for .doc format")
    void extractTextFromFile_DocFormat() {
        MockMultipartFile docFile = new MockMultipartFile(
                "file",
                "resume.doc",
                "application/msword",
                "content".getBytes());

        assertThrows(UnsupportedOperationException.class,
                () -> ResumeParserFactory.extractTextFromFile(docFile));
    }

    @Test
    @DisplayName("Should handle uppercase extension")
    void extractTextFromFile_UppercaseExtension() throws IOException {
        String content = "Resume content";
        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.TXT",
                "text/plain",
                content.getBytes());

        String result = ResumeParserFactory.extractTextFromFile(txtFile);

        assertEquals(content, result);
    }

    @Test
    @DisplayName("Should extract text from PDF file")
    void extractTextFromFile_Pdf() throws IOException {
        // Create a minimal valid PDF
        byte[] pdfBytes = createMinimalPdf();
        MockMultipartFile pdfFile = new MockMultipartFile(
                "file",
                "resume.pdf",
                "application/pdf",
                pdfBytes);

        String result = ResumeParserFactory.extractTextFromFile(pdfFile);
        assertNotNull(result);
    }

    @Test
    @DisplayName("Should extract text from DOCX file")
    void extractTextFromFile_Docx() throws IOException {
        // Create a minimal valid DOCX
        byte[] docxBytes = createMinimalDocx();
        MockMultipartFile docxFile = new MockMultipartFile(
                "file",
                "resume.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                docxBytes);

        String result = ResumeParserFactory.extractTextFromFile(docxFile);
        assertNotNull(result);
    }

    private byte[] createMinimalPdf() throws IOException {
        try (java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                org.apache.pdfbox.pdmodel.PDDocument doc = new org.apache.pdfbox.pdmodel.PDDocument()) {
            org.apache.pdfbox.pdmodel.PDPage page = new org.apache.pdfbox.pdmodel.PDPage();
            doc.addPage(page);
            doc.save(baos);
            return baos.toByteArray();
        }
    }

    private byte[] createMinimalDocx() throws IOException {
        try (java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                org.apache.poi.xwpf.usermodel.XWPFDocument doc = new org.apache.poi.xwpf.usermodel.XWPFDocument()) {
            doc.createParagraph().createRun().setText("Test content");
            doc.write(baos);
            return baos.toByteArray();
        }
    }
}
