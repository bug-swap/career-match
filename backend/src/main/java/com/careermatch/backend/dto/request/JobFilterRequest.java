package com.careermatch.backend.dto.request;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import java.math.BigDecimal;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class JobFilterRequest {
    private String category;
    private String location;
    private String jobType;
    private Boolean isRemote;
    private BigDecimal minSalary;
    private BigDecimal maxSalary;
    private String searchQuery;

    @Min(0)
    private int page = 0;

    @Min(1)
    @Max(100)
    private int size = 20;

    private String sortBy = "datePosted";
    private String sortOrder = "DESC";
}
