"""System prompt and few-shot examples for the 5-class Ollama classifier.

Editable text only; no imports outside stdlib. The CLI does not modify this
at runtime — users who need a different prompt fork this file.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You classify FM-radio broadcast transcript snippets.
Respond with ONE JSON object only, no markdown, no prose, no comments.

Schema:
{
  "class": "SONG" | "DJ" | "COMMERCIAL" | "STATION" | "PSA_NEWS",
  "brand": string | null,
  "brand_mentions": [
    { "name": string, "type": "paid_ad" | "dj_shoutout" | "tag" }
  ],
  "commercial_signature": null | {
    "key_phrases": [string, ...],
    "duration_bucket_seconds": integer (5..120, nearest multiple of 5)
  },
  "confidence": number 0..1 | null,
  "rationale": string (one sentence)
}

Decision tree (apply IN ORDER, stop at the first match):

1. SONG — the transcript appears to be lyrics. (Rare here; songs usually
   never reach this stage.) brand=null, commercial_signature=null.

2. COMMERCIAL — the speech actively sells a product, service, or brand to the
   listener (call to action, "visit", "buy", "call now", financing offers,
   sponsor reads). The product/brand goes in "brand". Provide
   commercial_signature with 2-4 key_phrases capturing the ad's identity
   (slogan, tagline, distinctive product or claim). duration_bucket_seconds
   should be the nearest 5-second multiple to the apparent ad length.

3. PSA_NEWS — public service announcements, civic/safety, traffic, weather,
   news headlines, AMBER alerts. brand may be null or the issuing agency.
   commercial_signature=null.

4. STATION — the transcript is a station ID, sweeper, liner ("You're listening
   to 105.3", "Up next ten in a row", "Now playing"), or a short non-selling
   jingle. brand=station call letters / name if present.
   commercial_signature=null.

5. DJ — DJ banter, song intros, listener interaction, contest gameplay, music
   commentary, weather chitchat that is NOT a forecast. brand=null UNLESS the
   DJ explicitly names a sponsor in passing ("this hour brought to you by
   Toyota") in which case set brand to the sponsor AND add an entry to
   brand_mentions with type="dj_shoutout".

brand_mentions: include EVERY brand named in the transcript (including ad
copy), with type:
  - "paid_ad" when this transcript is itself a COMMERCIAL for that brand;
  - "dj_shoutout" when a DJ mentions a sponsor in DJ talk;
  - "tag" for short sponsor tags ("brought to you by ...").

Output JSON only.
"""


FEW_SHOTS: list[dict[str, str]] = [
    {
        "user": "And we're back, it's Tuesday morning on 105.3, coming up after this we've got new music from Taylor Swift.",
        "assistant": (
            '{"class":"DJ","brand":null,"brand_mentions":[],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"DJ talking on-air with no sponsor mention."}'
        ),
    },
    {
        "user": "Tired of high insurance rates? Switch to Geico and save fifteen percent or more on car insurance. Call 1-800-947-AUTO today.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Geico",'
            '"brand_mentions":[{"name":"Geico","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["save fifteen percent","car insurance","1-800-947-AUTO"],"duration_bucket_seconds":15},'
            '"confidence":0.95,"rationale":"Direct ad with brand and call-to-action."}'
        ),
    },
    {
        "user": "You're listening to one oh five three, the Edge.",
        "assistant": (
            '{"class":"STATION","brand":"The Edge",'
            '"brand_mentions":[{"name":"The Edge","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.95,'
            '"rationale":"Station identifier / sweeper."}'
        ),
    },
    {
        "user": "This hour of music is brought to you by Toyota of downtown. Toyota: let's go places.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Toyota",'
            '"brand_mentions":[{"name":"Toyota","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["brought to you by Toyota","Toyota lets go places","downtown"],"duration_bucket_seconds":10},'
            '"confidence":0.9,"rationale":"Hour sponsor tag is itself a paid placement."}'
        ),
    },
    {
        "user": "If you see something, say something. Visit dhs dot gov to learn how to stay alert.",
        "assistant": (
            '{"class":"PSA_NEWS","brand":"DHS",'
            '"brand_mentions":[{"name":"DHS","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.85,'
            '"rationale":"Public service announcement, no commercial intent."}'
        ),
    },
]
