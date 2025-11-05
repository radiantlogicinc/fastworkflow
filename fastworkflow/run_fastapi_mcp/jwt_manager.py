"""
JWT Token Management for FastWorkflow FastAPI Service

Handles RSA key pair generation, JWT token creation and verification.
Keys are stored in ./jwt_keys/ directory.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from jose.constants import ALGORITHMS
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from fastworkflow.utils.logging import logger


# JWT Configuration (can be made configurable via env vars)
JWT_ALGORITHM = ALGORITHMS.RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30  # 30 days
JWT_ISSUER = "fastworkflow-api"
JWT_AUDIENCE = "fastworkflow-client"

# Key storage location (relative to project root)
KEYS_DIR = "./jwt_keys"
PRIVATE_KEY_PATH = os.path.join(KEYS_DIR, "private_key.pem")
PUBLIC_KEY_PATH = os.path.join(KEYS_DIR, "public_key.pem")

# In-memory cache for loaded keys
_private_key: Optional[str] = None
_public_key: Optional[str] = None

# Flag to control JWT verification behavior (set from CLI)
EXPECT_ENCRYPTED_JWT = True  # Default to secure mode


def ensure_keys_directory() -> None:
    """Create jwt_keys directory if it doesn't exist."""
    os.makedirs(KEYS_DIR, exist_ok=True)
    logger.info(f"JWT keys directory ensured at: {KEYS_DIR}")


def generate_rsa_key_pair() -> tuple[str, str]:
    """
    Generate a new RSA 2048-bit key pair.
    
    Returns:
        tuple[str, str]: (private_key_pem, public_key_pem)
    """
    logger.info("Generating new RSA 2048-bit key pair for JWT...")
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    # Serialize private key to PEM format
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')
    
    # Extract public key and serialize to PEM format
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode('utf-8')
    
    logger.info("RSA key pair generated successfully")
    return private_pem, public_pem


def save_keys_to_disk(private_pem: str, public_pem: str) -> None:
    """
    Save RSA keys to disk with appropriate permissions.
    
    Args:
        private_pem: Private key in PEM format
        public_pem: Public key in PEM format
    """
    ensure_keys_directory()
    
    # Save private key (mode 600 for security)
    with open(PRIVATE_KEY_PATH, 'w') as f:
        f.write(private_pem)
    os.chmod(PRIVATE_KEY_PATH, 0o600)
    logger.info(f"Private key saved to: {PRIVATE_KEY_PATH} (mode 600)")
    
    # Save public key (mode 644 is fine)
    with open(PUBLIC_KEY_PATH, 'w') as f:
        f.write(public_pem)
    os.chmod(PUBLIC_KEY_PATH, 0o644)
    logger.info(f"Public key saved to: {PUBLIC_KEY_PATH} (mode 644)")


def load_or_generate_keys() -> tuple[str, str]:
    """
    Load existing RSA keys from disk, or generate new ones if they don't exist.
    Caches keys in memory for performance.
    
    Returns:
        tuple[str, str]: (private_key_pem, public_key_pem)
    """
    global _private_key, _public_key
    
    # Return cached keys if available
    if _private_key and _public_key:
        return _private_key, _public_key
    
    # Try to load existing keys
    if os.path.exists(PRIVATE_KEY_PATH) and os.path.exists(PUBLIC_KEY_PATH):
        logger.info("Loading existing RSA keys from disk...")
        with open(PRIVATE_KEY_PATH, 'r') as f:
            _private_key = f.read()
        with open(PUBLIC_KEY_PATH, 'r') as f:
            _public_key = f.read()
        logger.info("RSA keys loaded successfully")
    else:
        # Generate and save new keys
        logger.info("No existing RSA keys found, generating new ones...")
        _private_key, _public_key = generate_rsa_key_pair()
        save_keys_to_disk(_private_key, _public_key)
    
    return _private_key, _public_key


def set_jwt_verification_mode(expect_encrypted: bool) -> None:
    """
    Configure JWT verification mode for trusted network scenarios.
    
    When expect_encrypted=False, JWT tokens are decoded without signature verification.
    This mode is ONLY suitable for trusted internal networks where JWT is used for
    data transport rather than security.
    
    WARNING: Disabling signature verification allows any client to forge tokens.
    Only use in controlled environments.
    """
    global EXPECT_ENCRYPTED_JWT
    EXPECT_ENCRYPTED_JWT = expect_encrypted
    if not expect_encrypted:
        logger.warning(
            "JWT signature verification DISABLED. "
            "Tokens will be accepted without cryptographic validation. "
            "Only use in trusted internal networks."
        )


def create_access_token(channel_id: str, user_id: Optional[str] = None, expires_days: int | None = None) -> str:
    """
    Create a JWT access token for a user.
    
    Behavior depends on EXPECT_ENCRYPTED_JWT flag:
    - If True: Creates a signed token using RSA algorithm
    - If False: Creates an unsigned token for trusted network use
    
    Args:
        channel_id: Channel identifier (required)
        user_id: User identifier (optional, will be included as uid claim if provided)
        expires_days: Optional custom expiration in days. If None, uses JWT_ACCESS_TOKEN_EXPIRE_MINUTES (default 60 minutes).
        
    Returns:
        str: Encoded JWT access token (signed or unsigned based on EXPECT_ENCRYPTED_JWT)
    """
    now = datetime.now(timezone.utc)
    if expires_days is not None:
        expire = now + timedelta(days=expires_days)
    else:
        expire = now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # JWT claims
    payload = {
        "sub": channel_id,  # Subject: the channel identifier
        "iat": int(now.timestamp()),  # Issued at
        "exp": int(expire.timestamp()),  # Expiration time
        "jti": f"{channel_id}_{int(now.timestamp())}",  # JWT ID (unique identifier)
        "type": "access",  # Token type
        "iss": JWT_ISSUER,  # Issuer
        "aud": JWT_AUDIENCE  # Audience
    }
    
    # Add optional user_id claim
    if user_id is not None:
        payload["uid"] = user_id
    
    if EXPECT_ENCRYPTED_JWT:
        # Secure mode: create signed token
        private_key, _ = load_or_generate_keys()
        token = jwt.encode(payload, private_key, algorithm=JWT_ALGORITHM)
        logger.debug(f"Created signed access token for channel_id: {channel_id}, user_id: {user_id}, expires: {expire.isoformat()}")
    else:
        # Trusted network mode: create unsigned token using HS256 with empty key
        # This creates a JWT that can be decoded without verification
        token = jwt.encode(payload, "", algorithm="HS256")
        logger.debug(f"Created unsigned access token for channel_id: {channel_id}, user_id: {user_id}, expires: {expire.isoformat()}")
    
    return token


def create_refresh_token(channel_id: str, user_id: Optional[str] = None) -> str:
    """
    Create a JWT refresh token for a user.
    
    Behavior depends on EXPECT_ENCRYPTED_JWT flag:
    - If True: Creates a signed token using RSA algorithm
    - If False: Creates an unsigned token for trusted network use
    
    Args:
        channel_id: Channel identifier (required)
        user_id: User identifier (optional, will be included as uid claim if provided)
        
    Returns:
        str: Encoded JWT refresh token (signed or unsigned based on EXPECT_ENCRYPTED_JWT)
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    # JWT claims
    payload = {
        "sub": channel_id,  # Subject: the channel identifier
        "iat": int(now.timestamp()),  # Issued at
        "exp": int(expire.timestamp()),  # Expiration time
        "jti": f"{channel_id}_{int(now.timestamp())}_refresh",  # JWT ID (unique identifier)
        "type": "refresh",  # Token type
        "iss": JWT_ISSUER,  # Issuer
        "aud": JWT_AUDIENCE  # Audience
    }
    
    # Add optional user_id claim
    if user_id is not None:
        payload["uid"] = user_id
    
    if EXPECT_ENCRYPTED_JWT:
        # Secure mode: create signed token
        private_key, _ = load_or_generate_keys()
        token = jwt.encode(payload, private_key, algorithm=JWT_ALGORITHM)
        logger.debug(f"Created signed refresh token for channel_id: {channel_id}, user_id: {user_id}, expires: {expire.isoformat()}")
    else:
        # Trusted network mode: create unsigned token using HS256 with empty key
        # This creates a JWT that can be decoded without verification
        token = jwt.encode(payload, "", algorithm="HS256")
        logger.debug(f"Created unsigned refresh token for channel_id: {channel_id}, user_id: {user_id}, expires: {expire.isoformat()}")
    
    return token


def verify_token(token: str, expected_type: str = "access") -> dict:
    # sourcery skip: extract-duplicate-method
    """
    Verify and decode a JWT token.
    
    Behavior depends on EXPECT_ENCRYPTED_JWT flag:
    - If True (default): Full cryptographic verification with signature check
    - If False: Extract payload without verification (trusted network mode)
    
    Args:
        token: JWT token string
        expected_type: Expected token type ("access" or "refresh")
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        JWTError: If token is invalid, expired, or type mismatch
    """
    if not EXPECT_ENCRYPTED_JWT:
        # Trusted network mode: decode without verification (accepts both unsigned and signed tokens)
        try:
            # Use unverified decoding - works for any JWT regardless of algorithm or signing
            payload = jwt.get_unverified_claims(token)
        except Exception as e:
            logger.warning(f"Token decoding failed: {e}")
            raise JWTError(f"Failed to decode token: {e}") from e
        
        # Manually check expiration even in unverified mode
        if exp_timestamp := payload.get("exp"):
            import time
            current_time = int(time.time())
            if exp_timestamp < current_time:
                logger.warning(f"Token expired: exp={exp_timestamp}, now={current_time}")
                raise JWTError("Token has expired")
        
        # Validate token type for consistency (outside try-except to allow JWTError to propagate)
        if payload.get("type") != expected_type:
            raise JWTError(f"Invalid token type: expected {expected_type}, got {payload.get('type')}")
        
        logger.debug(f"Token decoded (unverified mode): channel_id={payload.get('sub')}, type={expected_type}")
        return payload

    # Standard mode: full verification (existing code)
    _, public_key = load_or_generate_keys()

    try:
        # Decode and verify token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE
        )

        # Verify token type
        if payload.get("type") != expected_type:
            raise JWTError(f"Invalid token type: expected {expected_type}, got {payload.get('type')}")

        logger.debug(f"Token verified successfully: channel_id={payload.get('sub')}, type={expected_type}")
        return payload

    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        raise


def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Get the expiration time of a JWT token without full verification.
    Useful for debugging/logging.
    
    Args:
        token: JWT token string
        
    Returns:
        datetime: Expiration time in UTC, or None if token is invalid
    """
    try:
        # Decode without verification (just to inspect claims)
        payload = jwt.get_unverified_claims(token)
        if exp_timestamp := payload.get("exp"):
            return datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
    except Exception as e:
        logger.debug(f"Failed to get token expiry: {e}")
    return None

