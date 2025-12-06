package com.careermatch.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class CareerMatchApplication {
	public static void main(String[] args) {
        // Load environment variables from .env file

		SpringApplication.run(CareerMatchApplication.class, args);
	}
}