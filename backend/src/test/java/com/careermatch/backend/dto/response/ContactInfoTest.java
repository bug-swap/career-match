package com.careermatch.backend.dto.response;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ContactInfoTest {

    @Test
    void testBuilder() {
        ContactInfo info = ContactInfo.builder()
                .name("John")
                .email("john@test.com")
                .phone("123")
                .linkedin("linkedin.com/john")
                .github("github.com/john")
                .build();

        assertEquals("John", info.getName());
        assertEquals("john@test.com", info.getEmail());
        assertEquals("123", info.getPhone());
        assertEquals("linkedin.com/john", info.getLinkedin());
        assertEquals("github.com/john", info.getGithub());
    }

    @Test
    void testNoArgsAndSetters() {
        ContactInfo info = new ContactInfo();
        info.setName("Jane");
        info.setEmail("jane@test.com");
        info.setPhone("456");
        info.setLinkedin("li");
        info.setGithub("gh");

        assertEquals("Jane", info.getName());
        assertEquals("jane@test.com", info.getEmail());
        assertEquals("456", info.getPhone());
        assertEquals("li", info.getLinkedin());
        assertEquals("gh", info.getGithub());
    }

    @Test
    void testAllArgsConstructor() {
        ContactInfo info = new ContactInfo("A", "a@b.com", "789", "li", "gh");

        assertEquals("A", info.getName());
        assertEquals("a@b.com", info.getEmail());
        assertEquals("789", info.getPhone());
        assertEquals("li", info.getLinkedin());
        assertEquals("gh", info.getGithub());
    }
}
