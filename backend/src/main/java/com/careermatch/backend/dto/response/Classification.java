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
public class Classification {
    private String category;
    private Double confidence;

    @JsonProperty("top_3")
    private List<ClassificationDetail> top3;
}