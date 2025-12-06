package com.careermatch.backend.parse;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * Factory for parsing resumes using different strategies
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ResumeParserFactory {

    private final ParsingStrategyFactory parsingStrategyFactory;

    /**
     * Parse resume using the specified strategy
     * @param file the resume file
     * @param strategyType the parsing strategy type (AI or ML)
     * @return parsed Resume object
     */
    public Resume parse(MultipartFile file, ParsingStrategyType strategyType) {
        log.info("Parsing resume file: {} using strategy: {}",
                file.getOriginalFilename(),
                strategyType != null ? strategyType : "default");

        try {
            ResumeParsingStrategy strategy = parsingStrategyFactory.getStrategy(strategyType);
            return strategy.parse(file);
        } catch (Exception e) {
            log.error("Error during resume parsing", e);
            throw new RuntimeException("Failed to parse resume: " + e.getMessage(), e);
        }
    }

    /**
     * Parse resume using the default strategy
     * @param file the resume file
     * @return parsed Resume object
     */
    public Resume parse(MultipartFile file) {
        return parse(file, null);
    }

    /**
     * Extract text content from a resume file based on its format
     * @param file the uploaded resume file
     * @return extracted text content
     * @throws IOException if file reading fails
     */
    public static String extractTextFromFile(MultipartFile file) throws IOException {
        String filename = file.getOriginalFilename();
        if (filename == null) {
            throw new IllegalArgumentException("File name is null");
        }

        String extension = getFileExtension(filename).toLowerCase();
        log.info("Extracting text from file: {} with extension: {}", filename, extension);

        return switch (extension) {
            case "pdf" -> extractFromPdf(file.getInputStream());
            case "docx" -> extractFromDocx(file.getInputStream());
            case "doc" -> extractFromDoc(file.getInputStream());
            case "txt" -> extractFromTxt(file.getInputStream());
            default -> throw new IllegalArgumentException("Unsupported file format: " + extension);
        };
    }

    private static String getFileExtension(String filename) {
        int lastDotIndex = filename.lastIndexOf('.');
        if (lastDotIndex == -1) {
            throw new IllegalArgumentException("File has no extension: " + filename);
        }
        return filename.substring(lastDotIndex + 1);
    }

    private static String extractFromPdf(InputStream inputStream) throws IOException {
        try (PDDocument document = PDDocument.load(inputStream)) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(document);
        }
    }

    private static String extractFromDocx(InputStream inputStream) throws IOException {
        try (XWPFDocument document = new XWPFDocument(inputStream)) {
            StringBuilder text = new StringBuilder();
            for (XWPFParagraph paragraph : document.getParagraphs()) {
                text.append(paragraph.getText()).append("\n");
            }
            return text.toString();
        }
    }

    private static String extractFromDoc(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException("Legacy .doc format not yet supported. Please use .docx");
    }

    private static String extractFromTxt(InputStream inputStream) throws IOException {
        return new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
    }
}
