package com.careermatch.backend.repository;

import com.careermatch.backend.entity.Job;
import java.math.BigDecimal;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface JobRepository extends JpaRepository<Job, String> {

    @Query("""
            SELECT j FROM Job j
            WHERE (:category IS NULL OR LOWER(j.category) = LOWER(:category))
              AND (:location IS NULL OR LOWER(j.location) LIKE LOWER(CONCAT('%', :location, '%')))
              AND (:jobType IS NULL OR LOWER(j.jobType) = LOWER(:jobType))
              AND (:isRemote IS NULL OR j.isRemote = :isRemote)
              AND (:minSalary IS NULL OR j.minAmount >= :minSalary)
              AND (:maxSalary IS NULL OR j.maxAmount <= :maxSalary)
              AND (:searchQuery IS NULL OR (
                    LOWER(j.title) LIKE LOWER(CONCAT('%', :searchQuery, '%')) OR
                    LOWER(j.description) LIKE LOWER(CONCAT('%', :searchQuery, '%')) OR
                    LOWER(j.company) LIKE LOWER(CONCAT('%', :searchQuery, '%'))
              ))
            """)
    Page<Job> findWithFilters(
            @Param("category") String category,
            @Param("location") String location,
            @Param("jobType") String jobType,
            @Param("isRemote") Boolean isRemote,
            @Param("minSalary") BigDecimal minSalary,
            @Param("maxSalary") BigDecimal maxSalary,
            @Param("searchQuery") String searchQuery,
            Pageable pageable);
}
