package com.careermatch.backend.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.OffsetDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class JobWithScore {
    private String id;
    private String title;
    private String company;
    private String location;

    @JsonProperty("date_posted")
    private OffsetDateTime datePosted;

    @JsonProperty("job_type")
    private String jobType;

    @JsonProperty("is_remote")
    private Boolean isRemote;

    @JsonProperty("min_amount")
    private BigDecimal minAmount;

    @JsonProperty("max_amount")
    private BigDecimal maxAmount;

    private String currency;

    @JsonProperty("job_url")
    private String jobUrl;

    private String category;

    private Double score;
}
