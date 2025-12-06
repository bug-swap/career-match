package com.careermatch.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EntityInfo {
    @JsonProperty("job_titles")
    private List<String> jobTitles;

    private List<String> companies;

    @JsonProperty("work_dates")
    private List<String> workDates;

    private List<String> skills;
    private List<String> degrees;
    private List<String> majors;
    private List<String> institutions;

    @JsonProperty("graduation_years")
    private List<String> graduationYears;

    private String gpa;
    private List<String> certifications;
    private List<String> projects;
    private List<String> publications;
    private List<String> languages;
    private String summary;
}
