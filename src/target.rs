//! Base58Check + P2PKH helpers.
//!
//! The hot search loop only needs the 20-byte HASH160 of the target address, so
//! we decode the address once up-front and compare HASH160 values directly.

use anyhow::Result;
use sha2::{Digest, Sha256};

const B58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// HASH160 = RIPEMD160(SHA256(data)).
pub fn hash160(data: &[u8]) -> [u8; 20] {
    let h1 = Sha256::digest(data);
    let h2 = ripemd::Ripemd160::digest(h1);
    let mut out = [0u8; 20];
    out.copy_from_slice(&h2);
    out
}

fn sha256d(data: &[u8]) -> [u8; 32] {
    let h1 = Sha256::digest(data);
    let h2 = Sha256::digest(h1);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h2);
    out
}

/// Decode a Base58Check string into raw bytes.
pub fn base58_decode(s: &str) -> Result<Vec<u8>> {
    let bytes = s.as_bytes();
    let zeroes = bytes.iter().take_while(|&&c| c == b'1').count();
    let rest = &bytes[zeroes..];

    // big-endian accumulator, starts at 0
    let mut acc: Vec<u8> = vec![0u8];
    for &c in rest {
        let val = B58_ALPHABET
            .iter()
            .position(|&x| x == c)
            .ok_or_else(|| anyhow::anyhow!("invalid base58 character: '{}'", c as char))?
            as u32;
        let mut carry = val;
        for b in acc.iter_mut().rev() {
            let cur = (*b as u32) * 58 + carry;
            *b = (cur & 0xff) as u8;
            carry = cur >> 8;
        }
        while carry > 0 {
            acc.insert(0, (carry & 0xff) as u8);
            carry >>= 8;
        }
    }

    let mut out = vec![0u8; zeroes];
    let first_nonzero = acc.iter().position(|&b| b != 0).unwrap_or(acc.len());
    out.extend_from_slice(&acc[first_nonzero..]);
    Ok(out)
}

/// Encode bytes as a Base58Check string.
pub fn base58_encode(data: &[u8]) -> String {
    let zeroes = data.iter().take_while(|&&b| b == 0).count();
    let rest = &data[zeroes..];

    // little-endian base58 digits
    let mut digits: Vec<u8> = Vec::new();
    for &byte in rest {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            let cur = (*d as u32) * 256 + carry;
            *d = (cur % 58) as u8;
            carry = cur / 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }

    let mut s = String::with_capacity(zeroes + digits.len());
    for _ in 0..zeroes {
        s.push('1');
    }
    for &d in digits.iter().rev() {
        s.push(B58_ALPHABET[d as usize] as char);
    }
    s
}

/// Decode a P2PKH address (starts with '1') into its 20-byte HASH160.
pub fn address_to_hash160(addr: &str) -> Result<[u8; 20]> {
    let raw = base58_decode(addr)?;
    if raw.len() != 25 {
        anyhow::bail!("invalid address length {} (expected 25 bytes)", raw.len());
    }
    if raw[0] != 0x00 {
        anyhow::bail!("not a P2PKH address (version byte 0x{:02x})", raw[0]);
    }
    let checksum = &raw[21..25];
    let calc = &sha256d(&raw[0..21])[0..4];
    if checksum != calc {
        anyhow::bail!("address checksum mismatch");
    }
    let mut h = [0u8; 20];
    h.copy_from_slice(&raw[1..21]);
    Ok(h)
}

/// Encode a 20-byte HASH160 back into a P2PKH address.
pub fn hash160_to_address(h: &[u8; 20]) -> String {
    let mut payload = [0u8; 21];
    payload[0] = 0x00;
    payload[1..].copy_from_slice(h);
    let checksum = &sha256d(&payload)[0..4];

    let mut full = [0u8; 25];
    full[0..21].copy_from_slice(&payload);
    full[21..].copy_from_slice(checksum);
    base58_encode(&full)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base58_roundtrip() {
        let cases: &[&[u8]] = &[&[0u8], &[0, 0], &[1, 0], &[0xde, 0xad, 0xbe, 0xef], &[0u8; 20]];
        for c in cases {
            let enc = base58_encode(c);
            let dec = base58_decode(&enc).unwrap();
            assert_eq!(dec, *c, "roundtrip failed for {:?} -> {enc}", c);
        }
    }

    #[test]
    fn known_p2pkh() {
        // Private key 1 -> address 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
        let h = hash160(b"");
        let _ = h;
        let addr = "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH";
        let hash = address_to_hash160(addr).unwrap();
        let back = hash160_to_address(&hash);
        assert_eq!(back, addr);
    }
}
