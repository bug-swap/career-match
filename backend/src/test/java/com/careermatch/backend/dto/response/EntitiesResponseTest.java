package com.careermatch.backend.dto.response;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EntitiesResponseTest {

    @Test
    void testBuilder() {
        ContactInfo contact = ContactInfo.builder().name("John").build();
        EntityInfo entities = EntityInfo.builder().gpa("3.5").build();

        EntitiesResponse response = EntitiesResponse.builder()
                .success(true)
                .contact(contact)
                .entities(entities)
                .build();

        assertTrue(response.getSuccess());
        assertEquals("John", response.getContact().getName());
        assertEquals("3.5", response.getEntities().getGpa());
    }

    @Test
    void testNoArgsAndSetters() {
        EntitiesResponse response = new EntitiesResponse();
        ContactInfo contact = new ContactInfo();
        EntityInfo entities = new EntityInfo();

        response.setSuccess(false);
        response.setContact(contact);
        response.setEntities(entities);

        assertFalse(response.getSuccess());
        assertEquals(contact, response.getContact());
        assertEquals(entities, response.getEntities());
    }

    @Test
    void testAllArgsConstructor() {
        ContactInfo contact = new ContactInfo();
        EntityInfo entities = new EntityInfo();
        EntitiesResponse response = new EntitiesResponse(true, contact, entities);

        assertTrue(response.getSuccess());
        assertEquals(contact, response.getContact());
        assertEquals(entities, response.getEntities());
    }
}
