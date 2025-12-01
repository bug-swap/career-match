import os
import traceback
import pandas as pd
from datetime import datetime, date
from logging import Logger
import logging
from jobspy import scrape_jobs
from dotenv import load_dotenv
from database import DatabaseManager
import functions_framework

JOB_SITES_ROTATION = ["indeed", "linkedin"]
MINUTES_BETWEEN_CALLS = 7


def setup_logger() -> Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def safe_string_value(value, default_value=""):
    if pd.isna(value) or value is None:
        return default_value
    return str(value)


def safe_datetime_value(value):
    if pd.isna(value) or value is None or value is pd.NaT:
        return datetime.now().isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.isoformat()
        except:
            return datetime.now().isoformat()

    return datetime.now().isoformat()


def safe_float_value(value, default_value=None):
    if pd.isna(value) or value is None:
        return default_value
    try:
        return float(value)
    except:
        return default_value


def safe_boolean_value(value, default_value=False):
    if pd.isna(value) or value is None:
        return default_value
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "y"]
    try:
        return bool(value)
    except:
        return default_value


def get_current_scraping_site_by_time(logger):
    current_time = datetime.now()
    logger.info(f"Current time: {current_time}")

    minutes_since_epoch = int(current_time.timestamp() / 60)
    site_index = 1  # forced site (as in your version)
    selected_site = JOB_SITES_ROTATION[site_index]

    logger.info(f"Selected scraping site: {selected_site}")
    return selected_site


def scrape_jobs_from_site(location, position, results_limit, hours_old, site_name, logger):
    logger.info(
        f"Scraping jobs from {site_name}: location={location}, position={position}, "
        f"results={results_limit}, hours_old={hours_old}"
    )

    scraped = scrape_jobs(
        site_name=[site_name],
        search_term=position,
        google_search_term=f"{position} jobs in {location}",
        location=location,
        results_wanted=results_limit,
        hours_old=hours_old,
        country_indeed="USA",
    )

    logger.info(f"Scraped {len(scraped) if scraped is not None else 0} jobs")
    return scraped


def process_scraped_job(job_data_dict, category, logger):
    job_title = safe_string_value(job_data_dict.get("title", ""))

    if not job_title:
        logger.warning("Skipping job with empty title")
        return None

    job_record = {
        "id": safe_string_value(job_data_dict.get("id", "")),
        "title": job_title,
        "company": safe_string_value(job_data_dict.get("company", "")),
        "location": safe_string_value(job_data_dict.get("location", "")),
        "date_posted": safe_datetime_value(job_data_dict.get("date_posted", "")),
        "job_type": safe_string_value(job_data_dict.get("job_type", "")),
        "is_remote": safe_boolean_value(job_data_dict.get("is_remote", False)),
        "min_amount": safe_float_value(job_data_dict.get("min_amount")),
        "max_amount": safe_float_value(job_data_dict.get("max_amount")),
        "currency": safe_string_value(job_data_dict.get("currency", "")),
        "job_url": safe_string_value(job_data_dict.get("job_url", "")),
        "category": category,
    }

    if not job_record["id"] or not job_record["job_url"]:
        logger.warning("Skipping job missing required fields (id/url)")
        return None

    return job_record

def send_to_generate_embeddings(scrapped_jobs, logger):
    try:
        import requests
        api_url = os.getenv("EMBEDDING_API_URL")
        if not api_url:
            logger.error("EMBEDDING_API_URL not set in environment")
            return
        payload = {"jobs": scrapped_jobs}
        headers = {"Content-Type": "application/json"}  
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            logger.info(f"Sent jobs for embedding generation")
        else:
            logger.error(f"Failed to send jobs for embedding: {response.text}")
    except Exception as e:
        logger.error(f"Error sending jobs for embedding: {e}")
@functions_framework.http
def main(request):
    logger = setup_logger()
    database_manager = None

    try:
        data = request.get_json()
        if not data:
            return {"status": "error", "message": "No JSON provided"}, 400

        search_location = data.get("location")
        search_position = data.get("position")
        max_results = data.get("results_limit", 10)
        hours_old = data.get("hours_old", 24)
        category = data.get("category")

        if not search_location or not search_position or not category:
            return {"status": "error", "message": "location, position, and category required"}, 400

        logger.info(
            f"Starting scraper: location={search_location}, position={search_position}, category={category}, "
            f"results={max_results}, hours_old={hours_old}"
        )

        # current_site = get_current_scraping_site_by_time(logger)
        current_site = "indeed"

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_API_KEY")

        database_manager = DatabaseManager(
            logger=logger,
            project_url=supabase_url,
            api_key=supabase_key
        )
        database_manager.connect()

        scraped_jobs = scrape_jobs_from_site(
            search_location,
            search_position,
            max_results,
            hours_old,
            current_site,
            logger
        )

        if scraped_jobs is None or scraped_jobs.empty:
            return {
                "status": "success",
                "message": "No jobs found",
                "inserted": 0,
                "skipped": 0,
                "site": current_site,
            }

        inserted = 0
        skipped = 0
        jobs = []
        for _, row in scraped_jobs.iterrows():
            job_dict = row.to_dict()
            processed = process_scraped_job(job_dict, category, logger)
            if processed:
                jobs.append(processed)
            else:
                skipped += 1
                continue

            try:
                database_manager.insert("jobs", processed)
                inserted += 1
            except Exception as e:
                if "duplicate" in str(e).lower():
                    skipped += 1
                else:
                    skipped += 1
                    logger.error(f"Failed to insert: {e}")

        send_to_generate_embeddings(jobs, logger)

        return {
            "status": "success",
            "inserted": inserted,
            "skipped": skipped,
            "total_found": len(scraped_jobs),
            "site": current_site,
        }

    except Exception as e:
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}, 500

    finally:
        if database_manager:
            try:
                database_manager.disconnect()
            except:
                pass
