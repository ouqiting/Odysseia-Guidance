# -*- coding: utf-8 -*-
"""将备份文件上传到 Cloudflare R2（S3 兼容）对象存储。

纯 Python + 标准库实现 AWS SigV4 签名，零额外依赖、零镜像体积。
支持上传、列举、删除与远端按天保留清理。

环境变量（自动从 .env 读取）：
    R2_ACCOUNT_ID          必填，R2 账户 ID，用于拼接默认 endpoint
    R2_ENDPOINT            可选，覆盖默认 endpoint
                           默认 https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID       必填
    R2_SECRET_ACCESS_KEY   必填
    R2_BUCKET              必填
    R2_PREFIX              可选，远端对象前缀，如 odysseia/backup/
    R2_REGION              可选，默认 auto
    BACKUP_KEEP_DAYS       可选，远端保留天数，默认 3
"""

import os
import re
import sys
import logging
import datetime
import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANAGED_MARKERS = (".bak.", ".pgbak.")
_DATE_RE = re.compile(r"(\d{8})")
_R2_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# --- SigV4 签名 ------------------------------------------------------------

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    key = ("AWS4" + secret).encode("utf-8")
    key = _hmac(key, date_stamp)
    key = _hmac(key, region)
    key = _hmac(key, service)
    return _hmac(key, "aws4_request")


def _sigv4_headers(method, host, path, query, cfg, payload=b"", amz_date=None):
    """构造 SigV4 请求头。path 需已 URL 编码（保留 /），query 需已编码且按 key 排序。"""
    access_key, secret = cfg["access_key"], cfg["secret"]
    region, service = cfg["region"], "s3"
    if amz_date is None:
        amz_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_stamp = amz_date[:8]

    payload_hash = _sha256_hex(payload)
    canonical_uri = path
    canonical_querystring = query
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"

    canonical_request = "\n".join([
        method, canonical_uri, canonical_querystring,
        canonical_headers, signed_headers, payload_hash,
    ])
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, scope,
        _sha256_hex(canonical_request.encode("utf-8")),
    ])
    signature = _hmac(_signing_key(secret, date_stamp, region, service), string_to_sign).hex()

    return {
        "Host": host,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }


# --- 配置与 HTTP -----------------------------------------------------------

def _load_config():
    account = os.getenv("R2_ACCOUNT_ID", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.getenv("R2_BUCKET", "").strip()
    if not (account and access_key and secret and bucket):
        return None

    endpoint = os.getenv("R2_ENDPOINT", "").strip() or f"https://{account}.r2.cloudflarestorage.com"
    endpoint = endpoint.rstrip("/")
    prefix = os.getenv("R2_PREFIX", "").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret": secret,
        "bucket": bucket,
        "prefix": prefix,
        "region": os.getenv("R2_REGION", "auto").strip() or "auto",
    }


def _object_path(bucket: str, key: str) -> str:
    return "/" + bucket + "/" + key


def _s3_request(method, url, headers, data=b""):
    req = urllib.request.Request(url, data=data or None, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"S3 {method} {url} 失败: HTTP {e.code} {body}") from e


def _build_headers(method, path, cfg, query="", payload=b""):
    host = urllib.parse.urlparse(cfg["endpoint"]).netloc
    return _sigv4_headers(method, host, path, query, cfg, payload)


# --- 对象操作 ---------------------------------------------------------------

def put_object(cfg, key, data: bytes):
    path = urllib.parse.quote(_object_path(cfg["bucket"], key), safe="/-_.~")
    headers = _build_headers("PUT", path, cfg, payload=data)
    url = cfg["endpoint"] + path
    status, _ = _s3_request("PUT", url, headers, data)
    return status


def list_objects(cfg, prefix=""):
    objects = []
    continuation_token = None
    while True:
        query_parts = [("list-type", "2"), ("prefix", prefix)]
        if continuation_token:
            query_parts.append(("continuation-token", continuation_token))
        query_parts.sort(key=lambda kv: kv[0])
        query = "&".join(
            k + "=" + (v if k == "continuation-token" else urllib.parse.quote(v, safe="-_.~"))
            for k, v in query_parts
        )

        path = urllib.parse.quote(_object_path(cfg["bucket"], ""), safe="/-_.~")
        headers = _build_headers("GET", path, cfg, query=query)
        url = cfg["endpoint"] + path + "?" + query

        status, body = _s3_request("GET", url, headers)
        root = ET.fromstring(body.decode("utf-8", errors="replace"))
        for elem in root.iter():
            if _local(elem) == "Contents":
                key = size = None
                for child in elem:
                    tag = _local(child)
                    if tag == "Key":
                        key = child.text
                    elif tag == "Size":
                        size = child.text
                if key is not None:
                    objects.append({"Key": key, "Size": int(size or 0)})

        truncated = root.find(f"{_R2_NAMESPACE}IsTruncated")
        if truncated is None or truncated.text != "true":
            break
        token_el = root.find(f"{_R2_NAMESPACE}NextContinuationToken")
        continuation_token = token_el.text if token_el is not None else None
        if not continuation_token:
            break
    return objects


def delete_object(cfg, key):
    path = urllib.parse.quote(_object_path(cfg["bucket"], key), safe="/-_.~")
    headers = _build_headers("DELETE", path, cfg)
    url = cfg["endpoint"] + path
    status, _ = _s3_request("DELETE", url, headers)
    return status


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def delete_objects(cfg, keys) -> int:
    """批量删除对象（S3 DeleteObjects，等价于删除整个"文件夹"）。"""
    if not keys:
        return 0
    objects_xml = "".join(f"<Object><Key>{_xml_escape(k)}</Key></Object>" for k in keys)
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Delete><Quiet>true</Quiet>{objects_xml}</Delete>"
    )
    path = urllib.parse.quote(_object_path(cfg["bucket"], ""), safe="/-_.~")
    query = "delete="
    headers = _build_headers("POST", path, cfg, query=query, payload=body.encode("utf-8"))
    url = cfg["endpoint"] + path + "?delete"
    status, _ = _s3_request("POST", url, headers, body.encode("utf-8"))
    return len(keys)


def _local(elem):
    return elem.tag.split("}")[-1]


# --- 备份文件相关 ------------------------------------------------------------

def _is_managed(filename: str) -> bool:
    return any(marker in filename for marker in MANAGED_MARKERS)


def _allowed_dates(days: int) -> set:
    today = datetime.date.today()
    return {(today - datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range(max(1, days))}


def resolve_upload_source_dir() -> str:
    """返回需要上传的本地备份目录：容器内优先 /backup-output 挂载点。"""
    mount = os.getenv("BACKUP_OUTPUT_MOUNT_PATH", "/backup-output")
    if os.path.isdir(mount):
        return mount
    output = os.getenv("BACKUP_OUTPUT_DIR", "").strip()
    if output:
        if not os.path.isabs(output):
            output = os.path.join(BASE_DIR, output)
        return output
    return os.path.join(BASE_DIR, "src", "backup")


def upload_backups(source_dir: str = None, keep_days: int = None, prune: bool = True):
    """上传备份目录中的受管理备份文件到 R2，按日期文件夹存放。

    - 对象路径: {R2_PREFIX}/{YYYYMMDD}/{filename}
    - prune=True 时删除远端超过保留天数的整个日期文件夹
    - prune=False 时只上传不清理（备份失败时由调用方传入，避免误删旧备份）

    未配置 R2 环境变量时静默跳过。返回 {"uploaded": [...], "deleted": [...]}。
    """
    cfg = _load_config()
    if cfg is None:
        log.info("未配置 R2 环境变量，跳过云端上传。")
        return None

    if keep_days is None:
        keep_days = int(os.getenv("BACKUP_KEEP_DAYS", "3") or 3)
    if source_dir is None:
        source_dir = resolve_upload_source_dir()
    if not os.path.isdir(source_dir):
        log.warning(f"备份源目录不存在: {source_dir}")
        return None

    allowed = _allowed_dates(keep_days)
    uploaded, deleted = [], []

    # 1) 上传本地备份文件（按日期文件夹，覆盖已存在的同名对象，保证幂等）
    for filename in sorted(os.listdir(source_dir)):
        if not _is_managed(filename):
            continue
        local_path = os.path.join(source_dir, filename)
        if not os.path.isfile(local_path):
            continue
        if os.path.getsize(local_path) == 0:
            log.warning(f"跳过空备份文件（上传）：'{local_path}'")
            continue
        m = _DATE_RE.search(filename)
        date_folder = m.group(1) if m else "unknown"
        key = f"{cfg['prefix']}{date_folder}/{filename}"
        try:
            with open(local_path, "rb") as fh:
                data = fh.read()
            put_object(cfg, key, data)
            uploaded.append(filename)
            log.info(f"已上传到 R2: '{key}'")
        except Exception as e:
            log.error(f"上传 '{filename}' 到 R2 失败: {e}", exc_info=True)

    # 2) 远端清理：删除超过保留天数的整个日期文件夹
    if not prune:
        log.info("本次备份未完全成功，跳过 R2 远端过期清理（保留旧备份）。")
    else:
        try:
            remote_objects = list_objects(cfg, cfg["prefix"])
        except Exception as e:
            log.error(f"列举 R2 远端对象失败，跳过远端清理: {e}", exc_info=True)
            remote_objects = []

        expired_keys = []
        for obj in remote_objects:
            key = obj["Key"]
            rel = key[len(cfg["prefix"]):] if key.startswith(cfg["prefix"]) else key
            date_folder = rel.split("/", 1)[0]
            if _DATE_RE.fullmatch(date_folder) and date_folder not in allowed:
                expired_keys.append(key)

        if expired_keys:
            try:
                delete_objects(cfg, expired_keys)
                deleted = expired_keys
                log.info(f"已删除 R2 过期备份文件夹（{len(expired_keys)} 个对象）。")
            except Exception as e:
                log.error(f"批量删除 R2 过期备份失败: {e}", exc_info=True)

    log.info(f"R2 备份上传完成: 上传 {len(uploaded)} 个, 清理 {len(deleted)} 个。")
    return {"uploaded": uploaded, "deleted": deleted}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    result = upload_backups()
    sys.exit(0 if result is not None else 1)