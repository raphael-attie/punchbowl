from punchbowl.level0.decode import decode_ccsds_packets, write_level0, create_level0_from_packets

from prefect import flow, get_run_logger


@flow
def level0_core_flow(ccsds_path: str, output_path: str) -> None:
    logger = get_run_logger()

    logger.info("Beginning level 0 core flow")
    packet_contents = decode_ccsds_packets(ccsds_path)
    level0_file = create_level0_from_packets(packet_contents)
    write_level0(level0_file, output_path)
    logger.info("Ending level 0 core flow")
