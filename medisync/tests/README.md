# MediSync Test Suite

## Intensive Conversation Test

This test script simulates a realistic doctor-patient conversation with approximately 20+ dialogues from each side, testing the full MediSync system.

### What It Tests

**Doctor Agent:**
- Clinical note ingestion (21 notes)
- Hybrid search functionality
- Discovery API for contextual case finding
- Multiple patient case management

**Patient Agent:**
- Health diary logging (20 entries)
- Personal history retrieval
- Health insights using recommendation API
- Privacy-filtered data access

**System Features:**
- Qdrant vector storage and retrieval
- Dense and sparse embeddings
- Multi-tenant isolation (clinic_id filtering)
- Role-based access control
- End-to-end clinical workflow

### Test Scenario

The test simulates a complete 6-week treatment journey for a patient with a finger fracture:

**Phase 1: Initial Visit (Day 1)**
- Patient reports symptoms
- Doctor examines and documents findings

**Phase 2: Diagnosis (Day 5)**
- X-ray results confirm fracture
- Treatment plan established

**Phase 3: Patient Education**
- RICE protocol instruction
- Medication management

**Phase 4-5: Week 1-2 Progress**
- Side effect management
- Progress documentation
- Similar case searches

**Phase 6-7: Complications & Week 4**
- Minor complication (skin irritation)
- Follow-up X-ray
- Adjusted treatment plan

**Phase 8-10: Recovery (Week 6)**
- Bone healing confirmation
- Physical therapy exercises
- Final clearance for sports

**Phase 11-15: Knowledge Base Building**
- Additional case documentation
- Clinical searches and insights
- Prevention education
- Long-term care planning

### How to Run

```bash
# From the project root
python medisync/tests/test_intensive_conversation.py

# Or from the tests directory
cd medisync/tests
python test_intensive_conversation.py
```

### Expected Output

The test will display:
- Real-time conversation flow with emoji indicators
- Doctor thoughts, actions, and responses
- Patient logs and queries
- Search results in formatted tables
- Comprehensive summary report with statistics

### Success Metrics

- **Total Actions**: 41+ (20+ doctor, 20+ patient)
- **Doctor Ingests**: 21 clinical notes
- **Patient Diary Logs**: 20 entries
- **Searches**: Multiple semantic and hybrid searches
- **Success Rate**: Should be 100% with no errors

### Requirements

- All MediSync dependencies installed
- Qdrant cloud instance running and accessible
- SQLite database initialized
- Rich library for formatted output

### Troubleshooting

**Embedding API Errors:**
- The system automatically falls back to local FastEmbed
- First run will download the model (~532MB)

**Qdrant Connection Issues:**
- Check QDRANT_URL and QDRANT_API_KEY in .env
- Ensure cloud instance is running

**Database Errors:**
- Run `init_db()` to create necessary tables
- Check SQLite file permissions

### Test Output Example

```
═══ PHASE 1: Initial Visit (Day 1) ═══

👨‍⚕️  Doctor (Dialogue #1):
   add note P-001 Patient presents with right index finger pain...
   💭 Analyzing input: 'add note P-001...'
   ⚡ Ingesting note for P-001...
   🖥️  ✓ Saved (ID: abc123...)
   ╭─────────────────────────────╮
   │ Note recorded for **P-001** │
   ╰─────────────────────────────╯
```

### Test Coverage

- ✅ User registration and authentication
- ✅ Doctor note ingestion with embeddings
- ✅ Patient diary logging
- ✅ Hybrid search (dense + sparse vectors)
- ✅ History retrieval with filtering
- ✅ Health insights via recommendation API
- ✅ Multi-tenant data isolation
- ✅ Role-based access control
- ✅ Error handling and fallbacks
- ✅ Real-world clinical workflow

### Notes

- Test users are created with prefix `test_intensive`
- All data is stored in the real database (cleanup not automated)
- The test takes approximately 2-3 minutes to complete
- Pauses between actions simulate realistic interaction timing
