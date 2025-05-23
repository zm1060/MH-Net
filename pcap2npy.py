import argparse
import numpy as np
from scapy.all import *
import logging
import sys
import os
from tqdm import tqdm
import time as time_lib
import multiprocessing as mp
from itertools import islice
from typing import List, Tuple, Dict
import array

from utils import get_bytes_from_raw, get_bytes_from_raw_new
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CHUNK_SIZE = 1000  # Process packets in chunks for better performance

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {str(e)}")
            raise

def process_packet_chunk(packets: List) -> Tuple[List, List, int, int]:
    """Process a chunk of packets in parallel"""
    p_header_list = []
    p_payload_list = []
    pkt_count = 0
    failed_count = 0
    
    for pkt in packets:
        try:
            pkt_count += 1
            p_header = get_bytes_from_raw_new(pkt)
            p_payload = []
            if pkt.haslayer("Raw"):
                try:
                    _, p_payload = get_bytes_from_raw(hexdump(pkt["Raw"].load, dump=True))
                except:
                    p_payload = []
            
            p_header_list.append(p_header)
            p_payload_list.append(p_payload)
                
        except:
            failed_count += 1
            p_header_list.append([])
            p_payload_list.append([])
            
    return p_header_list, p_payload_list, pkt_count, failed_count

def process_pcap_file(file_path: str) -> Tuple[List, List, int, int]:
    """Process a single pcap file using multiprocessing"""
    with PcapReader(file_path) as packets:
        # Create packet chunks
        packet_chunks = []
        current_chunk = []
        
        for pkt in packets:
            current_chunk.append(pkt)
            if len(current_chunk) >= CHUNK_SIZE:
                packet_chunks.append(current_chunk)
                current_chunk = []
        
        if current_chunk:
            packet_chunks.append(current_chunk)
    
    # Process chunks in parallel
    with mp.Pool() as pool:
        results = list(tqdm(
            pool.imap(process_packet_chunk, packet_chunks),
            total=len(packet_chunks),
            desc="Processing packets"
        ))
    
    # Combine results
    p_header_list = []
    p_payload_list = []
    total_pkt_count = 0
    total_failed_count = 0
    
    for headers, payloads, pkt_count, failed_count in results:
        p_header_list.extend(headers)
        p_payload_list.extend(payloads)
        total_pkt_count += pkt_count
        total_failed_count += failed_count
    
    return p_header_list, p_payload_list, total_pkt_count, total_failed_count

def pcap2npy4ISCX(dir_path_dict: Dict[str, str], save_path_dict: Dict[str, str]):
    # First ensure all output directories exist
    for category in save_path_dict:
        ensure_dir_exists(save_path_dict[category])
    
    for category in dir_path_dict:
        dir_path = dir_path_dict[category]
        if not os.path.exists(dir_path):
            logging.error(f"Input directory does not exist: {dir_path}")
            continue
            
        file_list = [f for f in os.listdir(dir_path) if f.endswith('.pcap')]
        for file in tqdm(file_list, desc=f"Processing {category} files"):
            file_path = os.path.join(dir_path, file)
            logging.info(f'Processing file: {file_path}')
            
            try:
                start_time = time_lib.time()
                
                # Process the file
                p_header_list, p_payload_list, pkt_count, failed_count = process_pcap_file(file_path)
                
                end_time = time_lib.time()
                processing_time = end_time - start_time
                pps = pkt_count / processing_time if processing_time > 0 else 0
                logging.info(f"Processed {pkt_count} packets in {processing_time:.2f} seconds ({pps:.2f} packets/second)")
                
                if failed_count > 0:
                    logging.warning(f"Failed to process {failed_count} packets out of {pkt_count}")
                
                if pkt_count == 0:
                    logging.warning(f"No packets processed in {file_path}")
                    continue
                
                # Save results
                save_file = file[:-4] + 'npz'
                save_path = os.path.join(save_path_dict[category], save_file)
                logging.info(f'Saving processed data to: {save_path}')
                
                np.savez_compressed(save_path,
                                  header=np.array(p_header_list, dtype=object),
                                  payload=np.array(p_payload_list, dtype=object))
                
                logging.info(f"Successfully saved {save_path}")
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    elif opt.dataset == 'ciciot':
        config = CICIoTConfig()
    else:
        raise Exception('Dataset Error')

    try:
        pcap2npy4ISCX(dir_path_dict=config.DIR_PATH_DICT, save_path_dict=config.DIR_SAVE_DICT)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)