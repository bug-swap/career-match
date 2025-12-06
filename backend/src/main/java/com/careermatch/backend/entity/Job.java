package com.careermatch.backend.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.math.BigDecimal;
import java.time.OffsetDateTime;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "jobs")
public class Job {

    @Id
    private String id;

    private String title;
    private String company;
    private String location;
    private String category;

    @Column(name = "job_type")
    private String jobType;

    @Column(name = "is_remote")
    private Boolean isRemote;

    @Column(name = "min_amount")
    private BigDecimal minAmount;

    @Column(name = "max_amount")
    private BigDecimal maxAmount;

    private String currency;

    @Column(name = "job_url")
    private String jobUrl;

    @Column(name = "date_posted")
    private OffsetDateTime datePosted;

    @Column(columnDefinition = "TEXT")
    private String description;
}
