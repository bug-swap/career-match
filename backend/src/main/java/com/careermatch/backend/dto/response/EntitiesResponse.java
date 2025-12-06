package com.careermatch.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EntitiesResponse {
    private Boolean success;
    private ContactInfo contact;
    private EntityInfo entities;
}
