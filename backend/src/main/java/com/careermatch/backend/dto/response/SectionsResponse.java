package com.careermatch.backend.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Response DTO for Python ML service sections extraction
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SectionsResponse {

    private boolean success;
    private Sections sections;
    private Metadata metadata;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Sections {
        private List<Education> education;
        private List<Experience> experience;
        private List<Project> project;
        private List<Publication> publication;
        private Contact contact;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Education {
        private String degree;
        private String major;
        private String institution;
        private String location;
        private String date;
        private String gpa;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Experience {
        private String title;
        private String company;
        private String date;
        private Boolean isPresent;
        private String location;
        private List<String> responsibilities;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Project {
        private String name;
        private String url;
        private List<String> description;
        private String technologies;
        private String date;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Publication {
        private String title;
        private String year;
        private String doi;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Contact {
        private String name;
        private String email;
        private String phone;
        private String linkedin;
        private String github;
        private String website;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Metadata {
        @JsonProperty("section_count")
        private Integer sectionCount;

        @JsonProperty("total_items")
        private Integer totalItems;

        @JsonProperty("total_char_count")
        private Integer totalCharCount;

        @JsonProperty("processing_time_ms")
        private Long processingTimeMs;
    }
}