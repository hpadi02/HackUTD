from web3 import Web3
import json

# Connect to Ethereum/Polygon Blockchain
class BlockchainService:
    def __init__(self, rpc_url: str, contract_address: str = None, contract_abi: str = None):
        """
        Initialize the blockchain connection and optionally load a contract.
        :param rpc_url: The RPC URL for the blockchain (e.g., Infura, Alchemy).
        :param contract_address: The address of the deployed smart contract (optional).
        :param contract_abi: The ABI of the deployed smart contract (optional).
        """
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise ConnectionError("Failed to connect to the blockchain.")
        
        print(f"Connected to blockchain at {rpc_url}")

        self.contract = None
        if contract_address and contract_abi:
            self.contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=json.loads(contract_abi)
            )
            print("Contract loaded.")

    def create_wallet(self):
        """
        Generate a new wallet address and private key.
        """
        account = self.web3.eth.account.create()
        return {
            "address": account.address,
            "private_key": account.key.hex()
        }

    def get_wallet_balance(self, wallet_address: str):
        """
        Get the balance of a wallet in Ether.
        :param wallet_address: The address of the wallet.
        """
        try:
            checksum_address = Web3.to_checksum_address(wallet_address)
            balance_wei = self.web3.eth.get_balance(checksum_address)
            balance_eth = self.web3.from_wei(balance_wei, 'ether')
            return balance_eth
        except Exception as e:
            raise ValueError(f"Failed to fetch wallet balance: {e}")

    def send_transaction(self, private_key: str, to_address: str, amount_eth: float):
        """
        Send cryptocurrency from one wallet to another.
        :param private_key: The private key of the sender's wallet.
        :param to_address: The recipient's wallet address.
        :param amount_eth: The amount to send in Ether.
        """
        try:
            account = self.web3.eth.account.from_key(private_key)
            nonce = self.web3.eth.get_transaction_count(account.address)

            txn = {
                'nonce': nonce,
                'to': Web3.to_checksum_address(to_address),
                'value': self.web3.to_wei(amount_eth, 'ether'),
                'gas': 21000,
                'gasPrice': self.web3.to_wei('50', 'gwei'),
            }

            signed_txn = self.web3.eth.account.sign_transaction(txn, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            return self.web3.to_hex(tx_hash)

        except Exception as e:
            raise ValueError(f"Transaction failed: {e}")

    def get_transaction_status(self, tx_hash: str):
        """
        Get the status of a transaction.
        :param tx_hash: The transaction hash.
        """
        try:
            tx_receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            if tx_receipt and tx_receipt['status'] == 1:
                return "Success"
            return "Failed"
        except Exception as e:
            raise ValueError(f"Failed to fetch transaction status: {e}")
