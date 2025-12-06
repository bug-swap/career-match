package com.careermatch.backend;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class BackendApplicationTests {

	@Test
	void contextLoads() {
	}

	@Test
	void mainMethodRuns() {
		// Test the main method - it will start the application context
		// which is already loaded by @SpringBootTest, so just verify no exception
		CareerMatchApplication.main(new String[]{});
	}
}
